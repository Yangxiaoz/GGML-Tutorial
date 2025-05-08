
if (GGML_ALL_WARNINGS)
  if (NOT MSVC)
      set(cxx_flags
          # TODO(marella): Add other warnings.
          -Wpedantic
          -Wunused-variable
          -Wno-unused-function
          -Wno-multichar
      )
      add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
  endif()
endif()


# flags
ggml_add_compile_flags()

add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(common-ggml STATIC common-ggml.cpp)
target_link_libraries(common-ggml PRIVATE ggml)
target_include_directories(common-ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(custom-model STATIC custom-model.cpp)
target_link_libraries(custom-model PRIVATE ggml)
target_include_directories(custom-model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(demo-sgemm)
add_subdirectory(demo-split_graph)
