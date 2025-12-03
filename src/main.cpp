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

// Set to 1 to enable timing outputs, 0 to strip them.
#define ENABLE_TIMING 1

#if ENABLE_TIMING
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct TimingInfo {
    double decode_ms = 0.0;  // sum of cap.read() time
    double filter_ms = 0.0;  // sum of processFramePipeline() time
    double encode_ms = 0.0;  // sum of writer.write() time
    std::mutex decode_mu;
    std::mutex filter_mu;
    std::mutex encode_mu;
};
#endif

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
    // 影像容器（矩陣）
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(ksize, ksize), sigma);

    cv::Mat img32f, blurred32f;
    // 型別/亮度乘倍轉換
    img.convertTo(img32f, CV_32F);
    blurred.convertTo(blurred32f, CV_32F);
    
    // 原圖 - 模糊圖 = 細節 (Edges + High-Frequency)
    // 原圖 + strength * 細節 = 銳化結果
    cv::Mat out32f = img32f + strength * (img32f - blurred32f);
    cv::Mat out;
    out32f.convertTo(out, img.type());
    return out;
}

cv::Mat processFramePipeline(const cv::Mat &frame,
                             int ksize_denoising = 3,
                             int ksize_sharpening = 5) {
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

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

#if ENABLE_TIMING
    double decode_ms = 0.0;
    double filter_ms = 0.0;
    double encode_ms = 0.0;
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
        if (ok && !frame.empty()) {
            decode_ms += std::chrono::duration_cast<Ms>(td1 - td0).count();
        }
#endif
        if (!ok || frame.empty()) {
            break;
        }

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
    double total_wall_ms = std::chrono::duration_cast<Ms>(all_end - all_begin).count();
    double sum_ms = decode_ms + filter_ms + encode_ms;

    std::cout << "=== SERIAL timing (ms) ===\n";
    std::cout << "Decode: " << decode_ms << " ms\n";
    std::cout << "Filter: " << filter_ms << " ms\n";
    std::cout << "Encode: " << encode_ms << " ms\n";
    std::cout << "Sum(Decode+Filter+Encode): " << sum_ms << " ms\n";
    std::cout << "Wall time: " << total_wall_ms << " ms\n";
#endif
}

// ======================= Pipeline 1+1+1 =======================

void decode_stage_single(cv::VideoCapture &cap,
                         BoundedQueue<FramePacket> &decodeQueue
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
            Ms dt = std::chrono::duration_cast<Ms>(t1 - t0);
            std::lock_guard<std::mutex> lk(timing.decode_mu);
            timing.decode_ms += dt.count();
        }
#endif
        if (!ok || frame.empty()) {
            break;
        }
        decodeQueue.push(FramePacket(index, frame.clone(), false));
        ++index;
    }
    decodeQueue.push(FramePacket(-1, cv::Mat(), true));
}

void filter_stage_single(BoundedQueue<FramePacket> &decodeQueue,
                         BoundedQueue<FramePacket> &encodeQueue,
                         int ksize_denoising,
                         int ksize_sharpening
#if ENABLE_TIMING
                         , TimingInfo &timing
#endif
) {
    while (true) {
        FramePacket pkt;
        decodeQueue.pop(pkt);
        if (pkt.is_stop) {
            encodeQueue.push(FramePacket(-1, cv::Mat(), true));
            break;
        }

#if ENABLE_TIMING
        auto t0 = Clock::now();
#endif
        cv::Mat processed = processFramePipeline(pkt.frame, ksize_denoising, ksize_sharpening);
#if ENABLE_TIMING
        auto t1 = Clock::now();
        Ms dt = std::chrono::duration_cast<Ms>(t1 - t0);
        {
            std::lock_guard<std::mutex> lk(timing.filter_mu);
            timing.filter_ms += dt.count();
        }
#endif
        encodeQueue.push(FramePacket(pkt.index, processed, false));
    }
}

void encode_stage_generic(BoundedQueue<FramePacket> &encodeQueue,
                          const std::string &output_path,
                          double fps,
                          int width,
                          int height,
                          int expected_stop_tokens
#if ENABLE_TIMING
                          , TimingInfo &timing
#endif
) {
    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height)
    );
    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

    std::unordered_map<int, cv::Mat> pending;
    int next_index = 0;
    int stop_seen = 0;

    while (true) {
        FramePacket pkt;
        encodeQueue.pop(pkt);

        if (pkt.is_stop) {
            ++stop_seen;
            if (stop_seen >= expected_stop_tokens) {
                break;
            }
            continue;
        }

        pending[pkt.index] = pkt.frame;

        while (true) {
            auto it = pending.find(next_index);
            if (it == pending.end()) break;

#if ENABLE_TIMING
            auto t0 = Clock::now();
#endif
            writer.write(it->second);
#if ENABLE_TIMING
            auto t1 = Clock::now();
            Ms dt = std::chrono::duration_cast<Ms>(t1 - t0);
            {
                std::lock_guard<std::mutex> lk(timing.encode_mu);
                timing.encode_ms += dt.count();
            }
#endif
            pending.erase(it);
            ++next_index;
        }
    }

    std::cout << "Encoding finished. Total frames written: " << next_index << std::endl;
}

void pipeline_mode_process(const std::string &input_path,
                           const std::string &output_path,
                           int ksize_denoising = 3,
                           int ksize_sharpening = 5,
                           size_t queue_size = 8) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << input_path << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    BoundedQueue<FramePacket> decodeQueue(queue_size);
    BoundedQueue<FramePacket> encodeQueue(queue_size);

#if ENABLE_TIMING
    TimingInfo timing;
    auto all_begin = Clock::now();
#endif

    std::thread decoder([&](){
        decode_stage_single(cap, decodeQueue
#if ENABLE_TIMING
                            , timing
#endif
        );
    });

    std::thread filter([&](){
        filter_stage_single(decodeQueue, encodeQueue,
                            ksize_denoising, ksize_sharpening
#if ENABLE_TIMING
                            , timing
#endif
        );
    });

    std::thread encoder([&](){
        encode_stage_generic(encodeQueue, output_path,
                             fps, width, height, 1
#if ENABLE_TIMING
                             , timing
#endif
        );
    });

    decoder.join();
    filter.join();
    encoder.join();

#if ENABLE_TIMING
    auto all_end = Clock::now();
    double total_wall_ms = std::chrono::duration_cast<Ms>(all_end - all_begin).count();
    double sum_ms = timing.decode_ms + timing.filter_ms + timing.encode_ms;

    std::cout << "\n=== PIPELINE (1+1+1) timing (ms) ===\n";
    std::cout << "Decode: " << timing.decode_ms << " ms\n";
    std::cout << "Filter: " << timing.filter_ms << " ms\n";
    std::cout << "Encode: " << timing.encode_ms << " ms\n";
    std::cout << "Sum(Decode+Filter+Encode): " << sum_ms << " ms\n";
    std::cout << "Wall time: " << total_wall_ms << " ms\n";
#endif
}

// ======================= Pipeline 1+N+1 =======================

void decode_stage_multi(cv::VideoCapture &cap,
                        BoundedQueue<FramePacket> &decodeQueue,
                        int num_filter_workers
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
            Ms dt = std::chrono::duration_cast<Ms>(t1 - t0);
            std::lock_guard<std::mutex> lk(timing.decode_mu);
            timing.decode_ms += dt.count();
        }
#endif
        if (!ok || frame.empty()) break;

        decodeQueue.push(FramePacket(index, frame.clone(), false));
        ++index;
    }

    for (int i = 0; i < num_filter_workers; ++i) {
        decodeQueue.push(FramePacket(-1, cv::Mat(), true));
    }
}

void filter_stage_multi(BoundedQueue<FramePacket> &decodeQueue,
                        BoundedQueue<FramePacket> &encodeQueue,
                        int ksize_denoising,
                        int ksize_sharpening
#if ENABLE_TIMING
                        , TimingInfo &timing
#endif
) {
    while (true) {
        FramePacket pkt;
        decodeQueue.pop(pkt);
        if (pkt.is_stop) {
            encodeQueue.push(FramePacket(-1, cv::Mat(), true));
            break;
        }

#if ENABLE_TIMING
        auto t0 = Clock::now();
#endif
        cv::Mat processed = processFramePipeline(pkt.frame, ksize_denoising, ksize_sharpening);
#if ENABLE_TIMING
        auto t1 = Clock::now();
        Ms dt = std::chrono::duration_cast<Ms>(t1 - t0);
        {
            std::lock_guard<std::mutex> lk(timing.filter_mu);
            timing.filter_ms += dt.count();
        }
#endif
        encodeQueue.push(FramePacket(pkt.index, processed, false));
    }
}

void pipeline_multi_mode_process(const std::string &input_path,
                                 const std::string &output_path,
                                 int num_filter_workers = 8,
                                 int ksize_denoising = 3,
                                 int ksize_sharpening = 5,
                                 size_t queue_size = 8) {
    if (num_filter_workers < 1) {
        num_filter_workers = 1;
    }

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << input_path << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    BoundedQueue<FramePacket> decodeQueue(queue_size);
    BoundedQueue<FramePacket> encodeQueue(queue_size);

#if ENABLE_TIMING
    TimingInfo timing;
    auto all_begin = Clock::now();
#endif

    std::thread decoder([&](){
        decode_stage_multi(cap, decodeQueue, num_filter_workers
#if ENABLE_TIMING
                           , timing
#endif
        );
    });

    std::vector<std::thread> filter_threads;
    filter_threads.reserve(num_filter_workers);
    for (int i = 0; i < num_filter_workers; ++i) {
        filter_threads.emplace_back([&](){
            filter_stage_multi(decodeQueue, encodeQueue,
                               ksize_denoising, ksize_sharpening
#if ENABLE_TIMING
                               , timing
#endif
            );
        });
    }

    std::thread encoder([&](){
        encode_stage_generic(encodeQueue, output_path,
                             fps, width, height, num_filter_workers
#if ENABLE_TIMING
                             , timing
#endif
        );
    });

    decoder.join();
    for (auto &t : filter_threads) {
        t.join();
    }
    encoder.join();

#if ENABLE_TIMING
    auto all_end = Clock::now();
    double total_wall_ms = std::chrono::duration_cast<Ms>(all_end - all_begin).count();
    double sum_ms = timing.decode_ms + timing.filter_ms + timing.encode_ms;

    std::cout << "\n=== PIPELINE_MULTI (1+N+1) timing (ms) ===\n";
    std::cout << "Decode: " << timing.decode_ms << " ms\n";
    std::cout << "Filter: " << timing.filter_ms << " ms\n";
    std::cout << "Encode: " << timing.encode_ms << " ms\n";
    std::cout << "Sum(Decode+Filter+Encode): " << sum_ms << " ms\n";
    std::cout << "Wall time: " << total_wall_ms << " ms\n";
#endif
}

// ======================= Frame Parallel Mode =======================

void frame_mode_process(const std::string &input_path,
                        const std::string &output_path,
                        int num_workers = 4,
                        int ksize_denoising = 3,
                        int ksize_sharpening = 5) {
    if (num_workers < 1) {
        num_workers = 1;
    }

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << input_path << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

#if ENABLE_TIMING
    auto t_dec0 = Clock::now();
#endif
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        frames.push_back(frame.clone());
    }
#if ENABLE_TIMING
    auto t_dec1 = Clock::now();
    double decode_ms = std::chrono::duration_cast<Ms>(t_dec1 - t_dec0).count();
#endif

    std::vector<cv::Mat> processed(frames.size());

#if ENABLE_TIMING
    auto t_flt0 = Clock::now();
#endif
    if (num_workers == 1 || frames.empty()) {
        for (size_t i = 0; i < frames.size(); ++i) {
            processed[i] = processFramePipeline(frames[i], ksize_denoising, ksize_sharpening);
        }
    } else {
        size_t total = frames.size();
        size_t chunk = (total + num_workers - 1) / num_workers;
        std::vector<std::thread> workers;
        workers.reserve(num_workers);

        for (int w = 0; w < num_workers; ++w) {
            size_t start = w * chunk;
            if (start >= total) break;
            size_t end = std::min(start + chunk, total);
            workers.emplace_back([&, start, end](){
                for (size_t i = start; i < end; ++i) {
                    processed[i] = processFramePipeline(frames[i], ksize_denoising, ksize_sharpening);
                }
            });
        }
        for (auto &t : workers) {
            t.join();
        }
    }
#if ENABLE_TIMING
    auto t_flt1 = Clock::now();
    double filter_ms = std::chrono::duration_cast<Ms>(t_flt1 - t_flt0).count();
#endif

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

#if ENABLE_TIMING
    auto t_enc0 = Clock::now();
#endif
    for (const auto &f_out : processed) {
        writer.write(f_out);
    }
#if ENABLE_TIMING
    auto t_enc1 = Clock::now();
    double encode_ms = std::chrono::duration_cast<Ms>(t_enc1 - t_enc0).count();
    double sum_ms = decode_ms + filter_ms + encode_ms;

    std::cout << "\n=== FRAME-PARALLEL timing (ms) ===\n";
    std::cout << "Decode: " << decode_ms << " ms\n";
    std::cout << "Filter: " << filter_ms << " ms\n";
    std::cout << "Encode: " << encode_ms << " ms\n";
    std::cout << "Sum(Decode+Filter+Encode): " << sum_ms << " ms\n";
    std::cout << "Wall time: " << sum_ms << " ms\n";  // 解碼+處理+編碼是 sequential phases
#endif

    std::cout << "Frame-parallel processing finished, frames: " << frames.size() << std::endl;
}

// ======================= Tile Mode =======================

struct Tile {
    int x0, y0, x1, y1;
};

cv::Mat process_frame_tiled(const cv::Mat &frame,
                            int tiles_x,
                            int tiles_y,
                            int num_workers,
                            int ksize_denoising,
                            int ksize_sharpening) {
    cv::Mat output(frame.size(), frame.type());
    int h = frame.rows;
    int w = frame.cols;

    std::vector<Tile> tasks;
    tasks.reserve(tiles_x * tiles_y);
    int tile_w = (w + tiles_x - 1) / tiles_x;
    int tile_h = (h + tiles_y - 1) / tiles_y;

    for (int ty = 0; ty < tiles_y; ++ty) {
        int y0 = ty * tile_h;
        int y1 = std::min(y0 + tile_h, h);
        for (int tx = 0; tx < tiles_x; ++tx) {
            int x0 = tx * tile_w;
            int x1 = std::min(x0 + tile_w, w);
            if (x0 < x1 && y0 < y1) {
                tasks.push_back(Tile{x0,y0,x1,y1});
            }
        }
    }

    if (tasks.empty()) {
        frame.copyTo(output);
        return output;
    }

    if (num_workers < 1) {
        num_workers = 1;
    }
    int workers = std::min<int>(num_workers, tasks.size());

    std::atomic<size_t> next_task(0);

    auto worker_fn = [&]() {
        while (true) {
            size_t idx = next_task.fetch_add(1);
            if (idx >= tasks.size()) {
                break;
            }
            const Tile &t = tasks[idx];
            cv::Rect roi(t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0);
            cv::Mat tile_in = frame(roi);
            cv::Mat tile_out = processFramePipeline(tile_in, ksize_denoising, ksize_sharpening);
            tile_out.copyTo(output(roi));
        }
    };

    if (workers == 1) {
        worker_fn();
    } else {
        std::vector<std::thread> workers_vec;
        workers_vec.reserve(workers);
        for (int i = 0; i < workers; ++i) {
            workers_vec.emplace_back(worker_fn);
        }
        for (auto &t : workers_vec) {
            t.join();
        }
    }

    return output;
}

void tile_mode_process(const std::string &input_path,
                       const std::string &output_path,
                       int tiles_x,
                       int tiles_y,
                       int num_workers = 4,
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

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << output_path << std::endl;
        return;
    }

#if ENABLE_TIMING
    double decode_ms = 0.0;
    double filter_ms = 0.0;
    double encode_ms = 0.0;
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
        if (ok && !frame.empty()) {
            decode_ms += std::chrono::duration_cast<Ms>(td1 - td0).count();
        }
#endif
        if (!ok || frame.empty()) {
            break;
        }

#if ENABLE_TIMING
        auto tf0 = Clock::now();
#endif
        cv::Mat processed = process_frame_tiled(frame, tiles_x, tiles_y,
                                                num_workers,
                                                ksize_denoising, ksize_sharpening);
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

#if ENABLE_TIMING
    auto all_end = Clock::now();
    double total_wall_ms = std::chrono::duration_cast<Ms>(all_end - all_begin).count();
    double sum_ms = decode_ms + filter_ms + encode_ms;

    std::cout << "=== TILE mode timing (ms) ===\n";
    std::cout << "Decode: " << decode_ms << " ms\n";
    std::cout << "Filter: " << filter_ms << " ms\n";
    std::cout << "Encode: " << encode_ms << " ms\n";
    std::cout << "Sum(Decode+Filter+Encode): " << sum_ms << " ms\n";
    std::cout << "Wall time: " << total_wall_ms << " ms\n";
#endif

    std::cout << "Tile mode processing finished, frames: " << index << std::endl;
}

// ======================= Main & Mode Parsing =======================

enum class Mode {
    SERIAL,
    PIPELINE,
    PIPELINE_MULTI,
    FRAME,
    TILE
};

Mode parse_mode(const std::string &m) {
    if (m == "serial")         return Mode::SERIAL;
    if (m == "pipeline")       return Mode::PIPELINE;
    if (m == "pipeline_multi") return Mode::PIPELINE_MULTI;
    if (m == "frame")          return Mode::FRAME;
    if (m == "tile")           return Mode::TILE;
    return Mode::SERIAL;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0] << " <input.mp4> <output.mp4> <mode> [extra args]\n"
            << "  mode:\n"
            << "    serial\n"
            << "    pipeline\n"
            << "    pipeline_multi <num_workers>\n"
            << "    frame <num_workers>\n"
            << "    tile <tiles_x> <tiles_y> <num_workers>\n";
        return 1;
    }

    // Make OpenCV single-threaded so that all CPU parallelism comes from our std::thread.
    cv::setUseOptimized(true);
    cv::setNumThreads(1);
    std::cout << "OpenCV max threads = " << cv::getNumThreads() << std::endl;

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    std::string mode_str    = argv[3];
    Mode mode = parse_mode(mode_str);

    switch (mode) {
        case Mode::SERIAL:
            std::cout << "Running in SERIAL mode...\n";
            serial_mode_process(input_path, output_path);
            break;
        case Mode::PIPELINE:
            std::cout << "Running in PIPELINE (1+1+1) mode...\n";
            pipeline_mode_process(input_path, output_path);
            break;
        case Mode::PIPELINE_MULTI: {
            int workers = 4;
            if (argc >= 5) {
                workers = std::stoi(argv[4]);
            }
            std::cout << "Running in PIPELINE_MULTI (1+N+1) mode with "
                      << workers << " filter threads...\n";
            pipeline_multi_mode_process(input_path, output_path, workers);
            break;
        }
        case Mode::FRAME: {
            int workers = 4;
            if (argc >= 5) {
                workers = std::stoi(argv[4]);
            }
            std::cout << "Running in FRAME-parallel mode with "
                      << workers << " workers...\n";
            frame_mode_process(input_path, output_path, workers);
            break;
        }
        case Mode::TILE: {
            if (argc < 7) {
                std::cerr << "Tile mode requires: tiles_x tiles_y num_workers\n";
                return 1;
            }
            int tiles_x = std::stoi(argv[4]);
            int tiles_y = std::stoi(argv[5]);
            int workers = std::stoi(argv[6]);
            std::cout << "Running in TILE mode with "
                      << tiles_x << "x" << tiles_y
                      << " tiles and " << workers << " workers...\n";
            tile_mode_process(input_path, output_path,
                              tiles_x, tiles_y, workers);
            break;
        }
    }

    return 0;
}
