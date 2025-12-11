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
