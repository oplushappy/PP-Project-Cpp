cmake_minimum_required(VERSION 3.15)
project(VideoPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Conan 產生的 CMakeDeps：會有 opencvConfig.cmake / opencv-config.cmake 等
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(pp_app
    src/main3.cpp
)

target_link_libraries(pp_app
    PRIVATE
        opencv::opencv
        Threads::Threads
)
# =============================

add_executable(ffmpeg_bench
    src/ffmpeg.cpp
)

# ffmpeg.cpp 主要只用到 C++ 標準庫，通常不需要連結 OpenCV
# 但如果為了保險起見 (或之後你想擴充)，連結 Threads 是好習慣
target_link_libraries(ffmpeg_bench
    PRIVATE
        Threads::Threads
)
