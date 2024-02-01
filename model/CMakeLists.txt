find_package(yaml-cpp)

add_library(yolov8_seg SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8_seg.cpp)

target_include_directories(yolov8_seg PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
if(USE_TENSORRT)
    target_link_libraries(yolov8_seg framework_trt framework_onnx yaml-cpp)
else()
    target_link_libraries(yolov8_seg framework_onnx yaml-cpp)
endif()
target_link_libraries(yolov8_seg ${OpenCV_LIBS})

add_library(yolov8 SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/yolov8.cpp)

target_include_directories(yolov8 PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
if(USE_TENSORRT)
    target_link_libraries(yolov8 framework_trt framework_onnx yaml-cpp)
else()
    target_link_libraries(yolov8 framework_onnx yaml-cpp)
endif()
target_link_libraries(yolov8 ${OpenCV_LIBS})