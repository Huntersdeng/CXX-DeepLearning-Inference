add_executable(yolo_test ${CMAKE_CURRENT_SOURCE_DIR}/yolo_test.cpp)
target_include_directories(yolo_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(yolo_test PUBLIC yolo_det yolo_seg)
target_link_directories(yolo_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(ocr_test ${CMAKE_CURRENT_SOURCE_DIR}/ocr_test.cpp)
target_include_directories(ocr_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(ocr_test PUBLIC ctc attn dbnet) 
target_link_directories(ocr_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(sam_test ${CMAKE_CURRENT_SOURCE_DIR}/sam_test.cpp)
target_include_directories(sam_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(sam_test PUBLIC sam)
target_link_directories(sam_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(clip_test ${CMAKE_CURRENT_SOURCE_DIR}/clip_test.cpp)
target_include_directories(clip_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(clip_test PUBLIC clip)
target_link_directories(clip_test PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

# add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
# target_include_directories(test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
# target_link_libraries(test ${ONNXRUNTIME_LIBS})
# target_link_directories(test PUBLIC ${OpenCV_LIBS})