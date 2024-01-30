add_executable(yolov8_seg ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/base/cv_model.cpp)

target_include_directories(yolov8_seg PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
if(USE_TENSORRT)
    target_link_libraries(yolov8_seg framework_trt)
else()
    target_link_libraries(yolov8_seg framework_onnx)
endif()
target_link_directories(yolov8_seg PUBLIC  ${OpenCV_LIBS})