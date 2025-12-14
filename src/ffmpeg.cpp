#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib> // for std::system
#include <cstdio>
#include <filesystem> // C++17

// 若你的編譯器不支援 C++17，可以拿掉 filesystem 相關檢查
namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// 執行單次 FFmpeg 指令並回傳執行時間 (秒)
double run_ffmpeg_test(const std::string& input, const std::string& output, int threads) {
    // 1. 建構 Filter Chain (與 Python 版本一致)
    // Brightness(x2) -> Median(r=1) -> Unsharp(5x5, str=1.5)
    std::string filter_chain = "colorlevels=rimax=0.5:gimax=0.5:bimax=0.5,median=radius=1,unsharp=5:5:1.5:5:5:0.0";

    // 2. 建構指令
    // -y: 覆蓋輸出
    // -threads N: 設定執行緒數
    // -c:a copy: 複製音訊
    // -hide_banner -loglevel error: 減少輸出雜訊
    std::string cmd = "ffmpeg -hide_banner -loglevel error -y";
    cmd += " -threads " + std::to_string(threads);
    cmd += " -i \"" + input + "\"";
    cmd += " -vf \"" + filter_chain + "\"";
    cmd += " -c:v libx264 -preset medium -c:a copy";
    cmd += " \"" + output + "\"";

    // 3. 執行並計時
    auto start = Clock::now();
    
    int ret = std::system(cmd.c_str());
    
    auto end = Clock::now();
    
    if (ret != 0) {
        std::cerr << "[Error] FFmpeg returned non-zero code: " << ret << std::endl;
        return -1.0;
    }

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int main(int argc, char** argv) {
    // 設定預設路徑，也可透過參數傳入
    std::string input_path = "videos/full.mp4";
    std::string output_base = "output/output_ffmpeg";

    if (argc > 1) input_path = argv[1];

    // 簡單檢查檔案是否存在
    if (!fs::exists(input_path)) {
        std::cerr << "Error: Input file not found at " << input_path << std::endl;
        return 1;
    }

    // 確保輸出目錄存在
    fs::create_directories("output");

    std::vector<int> thread_counts = {1, 2, 4, 8, 16};

    std::cout << "===========================================" << std::endl;
    std::cout << " Running FFmpeg Benchmark (C++ Wrapper) " << std::endl;
    std::cout << " Input: " << input_path << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Threads\t\tTime (seconds)" << std::endl;
    std::cout << "-------\t\t--------------" << std::endl;

    for (int t : thread_counts) {
        std::string output_path = output_base + "_t" + std::to_string(t) + ".mp4";
        
        std::cout << "Testing " << t << " threads..." << std::flush; // flush 確保文字先印出來
        
        double seconds = run_ffmpeg_test(input_path, output_path, t);
        
        // 清除剛剛印的 "Testing..." 行，改印結果 (為了美觀，非必要)
        std::cout << "\r" << t << "\t\t";
        if (seconds < 0) {
            std::cout << "FAILED" << std::endl;
        } else {
            std::cout << seconds << " s" << std::endl;
        }
    }

    std::cout << "===========================================" << std::endl;
    
    // 跑一次 Auto (FFmpeg 預設)
    std::cout << "Auto\t\t";
    std::string cmd_auto = "ffmpeg -hide_banner -loglevel error -y -i \"" + input_path + 
                           "\" -vf \"colorlevels=rimax=0.5:gimax=0.5:bimax=0.5,median=radius=1,unsharp=5:5:1.5:5:5:0.0\"" + 
                           " -c:v libx264 -preset medium -c:a copy output/output_ffmpeg_auto.mp4";
    
    auto start = Clock::now();
    std::system(cmd_auto.c_str());
    auto end = Clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << duration.count() << " s" << std::endl;

    return 0;
}