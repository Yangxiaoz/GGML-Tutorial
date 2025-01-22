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


#add demos:

add_subdirectory(demo-sgemm)


