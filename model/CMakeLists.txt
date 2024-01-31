find_package(yaml-cpp)

add_library(yolov8_seg SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8_seg.cpp)

target_include_directories(yolov8_seg PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
if(USE_TENSORRT)
    target_link_libraries(yolov8_seg framework_trt yaml-cpp)
else()
    target_link_libraries(yolov8_seg framework_onnx yaml-cpp)
endif()
target_link_directories(yolov8_seg PUBLIC  ${OpenCV_LIBS})