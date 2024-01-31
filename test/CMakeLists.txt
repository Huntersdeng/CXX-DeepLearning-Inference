add_executable(yolov8_test ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8_test.cpp)
target_include_directories(yolov8_test PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(yolov8_test PUBLIC yolov8_seg)