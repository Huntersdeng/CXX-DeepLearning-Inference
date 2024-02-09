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
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

if(USE_TENSORRT)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
  get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
  message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

  # TensorRT
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)
  else()
    set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
    set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)
  endif()

  message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
  message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

  list(APPEND INCLUDE_DIRS
          ${CUDA_INCLUDE_DIRS}
          ${OpenCV_INCLUDE_DIRS}
          ${TensorRT_INCLUDE_DIRS}
          )
  
else()
  list(APPEND INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  )
endif()

set(ONNXRUNTIME_LIBS /usr/lib/libonnxruntime.so)
set(Clipper_LIBS /usr/lib/x86_64-linux-gnu/libpolyclipping.so)

add_subdirectory(framework)
add_subdirectory(model)
add_subdirectory(test)