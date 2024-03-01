find_package(ZLIB REQUIRED)

add_library(clip SHARED ${CMAKE_CURRENT_SOURCE_DIR}/text_tokenizer.cpp 
                        ${CMAKE_CURRENT_SOURCE_DIR}/image_encoder.cpp 
                        ${CMAKE_CURRENT_SOURCE_DIR}/text_encoder.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/clip.cpp)
target_include_directories(clip PUBLIC 
  ${CMAKE_SOURCE_DIR}
  ${ICU_INCLUDE_DIRS}
  ${INCLUDE_DIRS}
  ${PROJECT_BINARY_DIR})
target_link_libraries(clip base_model yaml-cpp)
target_link_libraries(clip ${ZLIB_LIBRARIES} ${ICU_LIBRARIES} ${OpenCV_LIBS})