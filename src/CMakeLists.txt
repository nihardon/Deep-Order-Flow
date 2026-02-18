cmake_minimum_required(VERSION 3.14)
project(GraphHFTEngine)

# 1. Set C++ Standard (C++17 is good for modern math)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Optimization Flags (Make it fast)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Apple Silicon Specific Optimizations (NEON / Accelerate)
if(APPLE)
    add_compile_options(-O3 -march=native -ffast-math)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
endif()

# 3. Define the Executable
add_executable(hft_engine main.cpp)

# 4. Link Libraries
if(APPLE)
    target_link_libraries(hft_engine PRIVATE ${ACCELERATE_FRAMEWORK})
endif()