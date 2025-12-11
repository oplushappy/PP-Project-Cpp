#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <string>
#include <future>
#include <functional>
#include <algorithm>

// Set to 1 to enable timing outputs, 0 to strip them.
#define ENABLE_TIMING 1

#if ENABLE_TIMING
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct TimingInfo {
    double decode_ms = 0.0;
    double filter_ms = 0.0;
    double encode_ms = 0.0;
    std::mutex decode_mu;
    std::mutex filter_mu;
    std::mutex encode_mu;
};
#endif

// ======================= Thread Pool =======================
// 通用的執行緒池，避免反覆建立執行緒的開銷
class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

    // 加入任務，回傳 std::future 以便等待結果
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
            
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// ======================= Basic Structures =======================

struct FramePacket {
    int index;
    cv::Mat frame;
    bool is_stop;

    FramePacket() : index(-1), is_stop(false) {}
    FramePacket(int idx, const cv::Mat &f, bool stop = false)
        : index(idx), frame(f), is_stop(stop) {}
};

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    void push(const T &value) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [&]{ return queue_.size() < capacity_; });
        queue_.push(value);
        not_empty_.notify_one();
    }

    void pop(T &value) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [&]{ return !queue_.empty(); });
        value = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
    }

private:
    std::queue<T> queue_;
    size_t capacity_;
    std::mutex mtx_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};

// ======================= Filters (OpenCV) =======================

cv::Mat adjustBrightnessMul(const cv::Mat &img, double factor = 2.0) {
    cv::Mat out;
    img.convertTo(out, -1, factor, 0.0);
    return out;
}

cv::Mat medianFilterOpenCV(const cv::Mat &img, int ksize = 3) {
    cv::Mat out;
    cv::medianBlur(img, out, ksize);
    return out;
}

cv::Mat unsharpMaskOpenCV(const cv::Mat &img,
                          int ksize = 5,
                          double sigma = 1.0,
                          double strength = 1.5) {
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(ksize, ksize), sigma);

    cv::Mat img32f, blurred32f;
    img.convertTo(img32f, CV_32F);
    blurred.convertTo(blurred32f, CV_32F);
    
    cv::Mat out32f = img32f + strength * (img32f - blurred32f);
    cv::Mat out;
    out32f.convertTo(out, img.type());
    return out;
}

cv::Mat processFramePipeline(const cv::Mat &frame,
                             int ksize_denoising = 3,
                             int ksize_sharpening = 5) {
    // 簡單的 Pipeline
    cv::Mat out = adjustBrightnessMul(frame, 2.0);
    out = medianFilterOpenCV(out, ksize_denoising);
    out = unsharpMaskOpenCV(out, ksize_sharpening, 1.0, 1.5);
    return out;
}

// ======================= Serial Mode =======================

void serial_mode_process(const std::string &input_path,
                         const std::string &output_path,
                         int ksize_denoising = 3,
                         int ksize_sharpening = 5) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << input_path << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

#if ENABLE_TIMING
    double decode_ms = 0.0, filter_ms = 0.0, encode_ms = 0.0;
    auto all_begin = Clock::now();
#endif

    cv::Mat frame;
    int index = 0;

    while (true) {
#if ENABLE_TIMING
        auto td0 = Clock::now();
#endif
        bool ok = cap.read(frame);
#if ENABLE_TIMING
        auto td1 = Clock::now();
        if (ok && !frame.empty()) decode_ms += std::chrono::duration_cast<Ms>(td1 - td0).count();
#endif
        if (!ok || frame.empty()) break;

#if ENABLE_TIMING
        auto tf0 = Clock::now();
#endif
        cv::Mat processed = processFramePipeline(frame, ksize_denoising, ksize_sharpening);
#if ENABLE_TIMING
        auto tf1 = Clock::now();
        filter_ms += std::chrono::duration_cast<Ms>(tf1 - tf0).count();
#endif

#if ENABLE_TIMING
        auto te0 = Clock::now();
#endif
        writer.write(processed);
#if ENABLE_TIMING
        auto te1 = Clock::now();
        encode_ms += std::chrono::duration_cast<Ms>(te1 - te0).count();
#endif
        ++index;
    }
    std::cout << "Serial processing finished, frames: " << index << std::endl;

#if ENABLE_TIMING
    auto all_end = Clock::now();
    std::cout << "=== SERIAL timing (ms) ===\n";
    std::cout << "Decode: " << decode_ms << " Filter: " << filter_ms << " Encode: " << encode_ms << "\n";
    std::cout << "Wall time: " << std::chrono::duration_cast<Ms>(all_end - all_begin).count() << " ms\n";
#endif
}

// ======================= Pipeline Stages =======================

void decode_stage_generic(cv::VideoCapture &cap,
                          BoundedQueue<FramePacket> &outQueue,
                          int num_stop_tokens
#if ENABLE_TIMING
                          , TimingInfo &timing
#endif
) {
    int index = 0;
    cv::Mat frame;
    while (true) {
#if ENABLE_TIMING
        auto t0 = Clock::now();
#endif
        bool ok = cap.read(frame);
#if ENABLE_TIMING
        auto t1 = Clock::now();
        if (ok && !frame.empty()) {
            std::lock_guard<std::mutex> lk(timing.decode_mu);
            timing.decode_ms += std::chrono::duration_cast<Ms>(t1 - t0).count();
        }
#endif
        if (!ok || frame.empty()) break;

        // 【關鍵優化】必須使用 clone()，因為 cap.read 重用 buffer
        // 若不 clone，pipeline 後面的 filter 可能會讀到被新 frame 覆蓋的髒資料
        outQueue.push(FramePacket(index, frame.clone(), false));
        ++index;
    }

    // 發送停止信號
    for (int i = 0; i < num_stop_tokens; ++i) {
        outQueue.push(FramePacket(-1, cv::Mat(), true));
    }
}

void filter_stage_generic(BoundedQueue<FramePacket> &inQueue,
                          BoundedQueue<FramePacket> &outQueue,
                          int ksize_denoising,
                          int ksize_sharpening
#if ENABLE_TIMING
                          , TimingInfo &timing
#endif
) {
    while (true) {
        FramePacket pkt;
        inQueue.pop(pkt);
        if (pkt.is_stop) {
            outQueue.push(FramePacket(-1, cv::Mat(), true));
            break;
        }

#if ENABLE_TIMING
        auto t0 = Clock::now();
#endif
        cv::Mat processed = processFramePipeline(pkt.frame, ksize_denoising, ksize_sharpening);
#if ENABLE_TIMING
        auto t1 = Clock::now();
        {
            std::lock_guard<std::mutex> lk(timing.filter_mu);
            timing.filter_ms += std::chrono::duration_cast<Ms>(t1 - t0).count();
        }
#endif
        outQueue.push(FramePacket(pkt.index, processed, false));
    }
}

void encode_stage_generic(BoundedQueue<FramePacket> &inQueue,
                          const std::string &output_path,
                          double fps, int width, int height,
                          int expected_stop_tokens
#if ENABLE_TIMING
                          , TimingInfo &timing
#endif
) {
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

    std::unordered_map<int, cv::Mat> pending;
    int next_index = 0;
    int stop_seen = 0;

    while (true) {
        FramePacket pkt;
        inQueue.pop(pkt);

        if (pkt.is_stop) {
            ++stop_seen;
            if (stop_seen >= expected_stop_tokens) break;
            continue;
        }

        // 重新排序邏輯
        pending[pkt.index] = pkt.frame;
        while (pending.count(next_index)) {
#if ENABLE_TIMING
            auto t0 = Clock::now();
#endif
            writer.write(pending[next_index]);
#if ENABLE_TIMING
            auto t1 = Clock::now();
            {
                std::lock_guard<std::mutex> lk(timing.encode_mu);
                timing.encode_ms += std::chrono::duration_cast<Ms>(t1 - t0).count();
            }
#endif
            pending.erase(next_index);
            ++next_index;
        }
    }
    std::cout << "Encoding finished. Frames: " << next_index << std::endl;
}

// ======================= Pipeline Wrappers =======================

void pipeline_mode_process(const std::string &input_path, const std::string &output_path, int workers) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    BoundedQueue<FramePacket> decodeQueue(8);
    BoundedQueue<FramePacket> encodeQueue(8);

#if ENABLE_TIMING
    TimingInfo timing;
    auto all_begin = Clock::now();
#endif

    std::thread decoder([&](){
        decode_stage_generic(cap, decodeQueue, workers
#if ENABLE_TIMING
            , timing
#endif
        );
    });

    std::vector<std::thread> filters;
    for(int i=0; i<workers; ++i) {
        filters.emplace_back([&](){
            filter_stage_generic(decodeQueue, encodeQueue, 3, 5
#if ENABLE_TIMING
                , timing
#endif
            );
        });
    }

    std::thread encoder([&](){
        encode_stage_generic(encodeQueue, output_path, cap.get(cv::CAP_PROP_FPS),
                             (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT),
                             workers
#if ENABLE_TIMING
                             , timing
#endif
        );
    });

    decoder.join();
    for(auto &t : filters) t.join();
    encoder.join();

#if ENABLE_TIMING
    auto all_end = Clock::now();
    std::cout << "=== PIPELINE timing ===\nWall time: " 
              << std::chrono::duration_cast<Ms>(all_end - all_begin).count() << " ms\n";
#endif
}

// ======================= Frame Parallel (Batched) =======================

void frame_mode_process(const std::string &input_path,
                        const std::string &output_path,
                        int num_workers) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width, height));
    
    // 【關鍵改進】使用 ThreadPool + Batch Processing 避免 OOM
    ThreadPool pool(num_workers);
    const int BATCH_SIZE = num_workers * 2; // 每次讀取兩倍於 worker 的量，保持 pipeline 滿載但記憶體可控

#if ENABLE_TIMING
    auto all_begin = Clock::now();
#endif

    std::vector<cv::Mat> batch_frames;
    batch_frames.reserve(BATCH_SIZE);
    
    cv::Mat frame;
    bool video_active = true;

    while (video_active) {
        // 1. Read Batch
        batch_frames.clear();
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (cap.read(frame) && !frame.empty()) {
                batch_frames.push_back(frame.clone());
            } else {
                video_active = false;
                break;
            }
        }

        if (batch_frames.empty()) break;

        // 2. Parallel Process Batch
        std::vector<std::future<void>> futures;
        std::vector<cv::Mat> processed_batch(batch_frames.size());

        for (size_t i = 0; i < batch_frames.size(); ++i) {
            futures.emplace_back(pool.enqueue([&, i] {
                processed_batch[i] = processFramePipeline(batch_frames[i]);
            }));
        }

        // Wait for all
        for (auto &f : futures) f.get();

        // 3. Write Batch
        for (const auto &p_frame : processed_batch) {
            writer.write(p_frame);
        }
    }

#if ENABLE_TIMING
    std::cout << "=== FRAME Mode (Batched) ===\nWall time: " 
              << std::chrono::duration_cast<Ms>(Clock::now() - all_begin).count() << " ms\n";
#endif
}

// ======================= Tile Mode (Overlapped) =======================

struct Tile {
    int x0, y0, x1, y1; // Target ROI
    int ex0, ey0, ex1, ey1; // Expanded ROI (with padding)
};

// 產生帶有 Padding 的 Tile 任務，解決邊界瑕疵
std::vector<Tile> generate_tiles(int w, int h, int tiles_x, int tiles_y, int padding) {
    std::vector<Tile> tasks;
    int tile_w = (w + tiles_x - 1) / tiles_x;
    int tile_h = (h + tiles_y - 1) / tiles_y;

    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int x0 = tx * tile_w;
            int x1 = std::min(x0 + tile_w, w);
            int y0 = ty * tile_h;
            int y1 = std::min(y0 + tile_h, h);

            if (x0 >= x1 || y0 >= y1) continue;

            // 計算擴充範圍 (Expanded ROI)，但不超出影像邊界
            int ex0 = std::max(0, x0 - padding);
            int ey0 = std::max(0, y0 - padding);
            int ex1 = std::min(w, x1 + padding);
            int ey1 = std::min(h, y1 + padding);

            tasks.push_back({x0, y0, x1, y1, ex0, ey0, ex1, ey1});
        }
    }
    return tasks;
}

void tile_mode_process(const std::string &input_path,
                       const std::string &output_path,
                       int tiles_x, int tiles_y, int num_workers) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('a','v','c','1'), cap.get(cv::CAP_PROP_FPS), cv::Size(w, h));

    // 【關鍵優化】建立一次 ThreadPool，重複使用，大幅提升效能
    ThreadPool pool(num_workers);
    
    // 計算需要的 Padding (通常是最大 filter kernel 半徑)
    // 這裡最大的 filter 是 5x5，所以半徑是 2，保險起見設 3 或 4
    const int PADDING = 4;
    auto tiles = generate_tiles(w, h, tiles_x, tiles_y, PADDING);

#if ENABLE_TIMING
    auto all_begin = Clock::now();
#endif

    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        cv::Mat output(frame.size(), frame.type());
        std::vector<std::future<void>> futures;

        for (const auto &t : tiles) {
            futures.emplace_back(pool.enqueue([&, t]() {
                // 1. 剪裁擴充區域 (Expanded ROI)
                cv::Mat tile_src = frame(cv::Rect(t.ex0, t.ey0, t.ex1 - t.ex0, t.ey1 - t.ey0));
                
                // 2. 處理
                cv::Mat tile_proc = processFramePipeline(tile_src);

                // 3. 計算有效區域在 tile_proc 內的位置
                // 因為我們擴充了左上角，所以有效區域的起點會有 offset
                int valid_x = t.x0 - t.ex0;
                int valid_y = t.y0 - t.ey0;
                int valid_w = t.x1 - t.x0;
                int valid_h = t.y1 - t.y0;

                // 4. 將有效區域貼回 Output
                tile_proc(cv::Rect(valid_x, valid_y, valid_w, valid_h))
                    .copyTo(output(cv::Rect(t.x0, t.y0, valid_w, valid_h)));
            }));
        }

        // 等待所有 Tile 完成
        for(auto &f : futures) f.get();
        writer.write(output);
    }

#if ENABLE_TIMING
    std::cout << "=== TILE Mode (Pool + Overlap) ===\nWall time: " 
              << std::chrono::duration_cast<Ms>(Clock::now() - all_begin).count() << " ms\n";
#endif
}

// ======================= Main =======================

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input> <output> <mode> [args...]\n"
                  << "Modes: serial, pipeline, pipeline_multi, frame, tile\n";
        return 1;
    }

    cv::setUseOptimized(true);
    cv::setNumThreads(1); // 讓我們的 ThreadPool 接管並行處理

    std::string input = argv[1];
    std::string output = argv[2];
    std::string mode = argv[3];

    if (mode == "serial") {
        serial_mode_process(input, output);
    } 
    else if (mode == "pipeline") {
        std::cout << "Pipeline (1+1+1)\n";
        pipeline_mode_process(input, output, 1);
    }
    else if (mode == "pipeline_multi") {
        int w = (argc >= 5) ? std::stoi(argv[4]) : 4;
        std::cout << "Pipeline Multi (1+" << w << "+1)\n";
        pipeline_mode_process(input, output, w);
    }
    else if (mode == "frame") {
        int w = (argc >= 5) ? std::stoi(argv[4]) : 4;
        std::cout << "Frame Parallel (Batched) with " << w << " workers\n";
        frame_mode_process(input, output, w);
    }
    else if (mode == "tile") {
        if (argc < 7) { std::cerr << "Need: tile <tx> <ty> <workers>\n"; return 1; }
        int tx = std::stoi(argv[4]);
        int ty = std::stoi(argv[5]);
        int w = std::stoi(argv[6]);
        std::cout << "Tile Mode (" << tx << "x" << ty << ") with " << w << " workers\n";
        tile_mode_process(input, output, tx, ty, w);
    }

    return 0;
}