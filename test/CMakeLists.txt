add_executable(yolov8_test ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_test.cpp)
target_include_directories(yolov8_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(yolov8_test PUBLIC yolov8_end2end yolov8_normal yolov8_seg)
target_link_directories(yolov8_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(ocr_test ${CMAKE_CURRENT_SOURCE_DIR}/ocr_test.cpp)
target_include_directories(ocr_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(ocr_test PUBLIC ctc attn)
target_link_directories(ocr_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_include_directories(test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(test ${ONNXRUNTIME_LIBS})
target_link_directories(test PUBLIC ${OpenCV_LIBS})