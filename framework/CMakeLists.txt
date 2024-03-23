configure_file(../common/config.h.in config.h)

if(USE_TENSORRT)
  add_library(framework_trt SHARED ${CMAKE_SOURCE_DIR}/common/common.cpp 
                                   ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/tensorrt.cpp)
  target_include_directories(framework_trt PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
  target_link_libraries(framework_trt nvinfer nvinfer_plugin ${CUDA_LIBRARIES})
  target_link_directories(framework_trt PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
endif()

if(USE_RKNN)
  add_library(framework_rknn SHARED ${CMAKE_SOURCE_DIR}/common/common.cpp 
                                   ${CMAKE_CURRENT_SOURCE_DIR}/rknn/rknn.cpp)
  target_include_directories(framework_rknn PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
  target_link_libraries(framework_rknn ${RKNN_LIBS})
  target_link_directories(framework_rknn PUBLIC ${OpenCV_LIBS})
endif()

add_library(framework_onnx SHARED ${CMAKE_SOURCE_DIR}/common/common.cpp 
                                  ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.cpp)
target_include_directories(framework_onnx PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(framework_onnx ${ONNXRUNTIME_LIBS})
target_link_directories(framework_onnx PUBLIC ${OpenCV_LIBS})
