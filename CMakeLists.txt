cmake_minimum_required(VERSION 3.8)

option(USE_TENSORRT "Compile tensorrt framework" ON)
message(STATUS "USE_TENSORRT: ${USE_TENSORRT}\n")

if(USE_TENSORRT)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  project(model_zoo_cxx LANGUAGES CXX CUDA)
  option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
else()
  project(model_zoo_cxx)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O0")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

add_subdirectory(framework)