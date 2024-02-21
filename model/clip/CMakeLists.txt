find_package(ZLIB REQUIRED)

add_executable(clip_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/text_tokenizer.cpp)
target_include_directories(clip_test PUBLIC 
  ${CMAKE_SOURCE_DIR}
  ${ICU_INCLUDE_DIRS})
target_link_libraries(clip_test ${ZLIB_LIBRARIES} ${ICU_LIBRARIES})