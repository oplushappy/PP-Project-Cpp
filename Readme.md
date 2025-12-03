## Set Up
1. install conan (pip install conan)
2. conan profile detect --force
3. conan install . --output-folder=build --build=missing -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True
4. cmake --preset conan-release
5. cmake --build --preset conan-release
6. build/build/Release/pp_app video output mode thread_nums
    - `build/build/Release/pp_app videos/clip.mp4 out_serial.mp4 serial 8`
    - `build/build/Release/pp_app clip.mp4 out_multi.mp4 pipeline_multi 8`
    - `build/build/Release/pp_app videos/taipei.mp4 out_frame.mp4 frame 8`
    - `build/build/Release/pp_app videos/clip.mp4 out_tile.mp4 tile 2 4 8`