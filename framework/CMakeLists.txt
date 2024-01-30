configure_file(common/config.h.in config.h)

if(USE_TENSORRT)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
  get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
  message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

  # TensorRT
  set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
  set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

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

if(USE_TENSORRT)
  add_library(framework_trt SHARED ${CMAKE_CURRENT_SOURCE_DIR}/common/common.cpp 
                                   ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/tensorrt.cpp)
  target_include_directories(framework_trt PUBLIC ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
  target_link_libraries(framework_trt nvinfer nvinfer_plugin)
  target_link_directories(framework_trt PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
endif()

add_library(framework_onnx SHARED ${CMAKE_CURRENT_SOURCE_DIR}/common/common.cpp 
                                  ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.cpp)
target_include_directories(framework_onnx PUBLIC ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(framework_onnx ${ONNXRUNTIME_LIBS} ${OpenCV_LIBS})
