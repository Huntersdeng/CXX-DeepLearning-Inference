find_package(yaml-cpp)

add_library(base_model SHARED ${CMAKE_CURRENT_SOURCE_DIR}/base/model.cpp)
target_include_directories(base_model PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
if(USE_TENSORRT)
    target_link_libraries(base_model framework_trt framework_onnx)
else()
    target_link_libraries(base_model framework_onnx)
endif()

add_library(yolo_seg SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolo/yolo_seg.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/detection_model.cpp)
target_include_directories(yolo_seg PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(yolo_seg base_model yaml-cpp)
target_link_libraries(yolo_seg ${OpenCV_LIBS})

# add_library(yolo_end2end SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolo/yolo_e2e.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/detection_model.cpp)
# target_include_directories(yolo_end2end PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
# target_link_libraries(yolo_end2end base_model yaml-cpp)
# target_link_libraries(yolo_end2end ${OpenCV_LIBS})

add_library(yolo_det SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yolo/yolo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/detection_model.cpp)
target_include_directories(yolo_det PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(yolo_det base_model yaml-cpp)
target_link_libraries(yolo_det ${OpenCV_LIBS})

add_library(ctc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ocr/ctc.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/ocr_model.cpp)
target_include_directories(ctc PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(ctc base_model yaml-cpp)
target_link_libraries(ctc ${OpenCV_LIBS})

add_library(attn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ocr/attention.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/ocr_model.cpp)
target_include_directories(attn PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(attn base_model yaml-cpp)
target_link_libraries(attn ${OpenCV_LIBS})

add_library(dbnet SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ocr/dbnet.cpp ${CMAKE_CURRENT_SOURCE_DIR}/base/detection_model.cpp)
target_include_directories(dbnet PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(dbnet base_model yaml-cpp)
target_link_libraries(dbnet ${OpenCV_LIBS} ${Clipper_LIBS})

add_library(sam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/sam/image_encoder.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sam/mask_decoder.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sam/sam.cpp)
target_include_directories(sam PUBLIC ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(sam base_model yaml-cpp)
target_link_libraries(sam ${OpenCV_LIBS})

add_subdirectory(clip)