
#
# simple-backend

set(TEST_TARGET split_graph)
add_executable(${TEST_TARGET} split_graph.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml custom-model)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
