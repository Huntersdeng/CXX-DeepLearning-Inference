add_executable(yolov8_test ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_test.cpp)
target_include_directories(yolov8_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(yolov8_test PUBLIC yolov8_end2end yolov8_normal yolov8_seg)
target_link_directories(yolov8_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(crnn_test ${CMAKE_CURRENT_SOURCE_DIR}/crnn_test.cpp)
target_include_directories(crnn_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(crnn_test PUBLIC crnn)
target_link_directories(crnn_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})