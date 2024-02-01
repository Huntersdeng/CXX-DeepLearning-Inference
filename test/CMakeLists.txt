add_executable(yolov8_test ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8_test.cpp)
target_include_directories(yolov8_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(yolov8_test PUBLIC yolov8_seg nvinfer nvinfer_plugin)
target_link_directories(yolov8_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${ONNXRUNTIME_LIBS})

# add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
# target_include_directories(test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
# target_link_libraries(test PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})