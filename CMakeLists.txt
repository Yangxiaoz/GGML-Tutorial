cmake_minimum_required(VERSION 3.14) 
project("ggml_turorial" C CXX)
include(CheckIncludeFileCXX)# check the stand C lib

set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#set the type of cmake
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()



# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



if (MINGW)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    set(BUILD_SHARED_LIBS_DEFAULT ON)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ON)


if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()



#
# option list
#

# debug
option(ALL_WARNINGS       " enable all compiler warnings"         OFF)

# build
option(FATAL_WARNINGS     " enable -Werror flag"                  OFF)

# sanitizers
option(SANITIZE_THREAD    " enable thread sanitizer"              OFF)
option(SANITIZE_ADDRESS   " enable address sanitizer"             OFF)
option(ANITIZE_UNDEFINED  " enable undefined sanitizer"           OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# override ggml options
set(GGML_SANITIZE_THREAD    ${SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${ANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${FATAL_WARNINGS})

#
# build the library
#
if (NOT TARGET ggml)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# install
#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

##
# At the moment some compile definitions are placed within the ggml/src
# directory but not exported on the `ggml` target. This could be improved by
# determining _precisely_ which defines are necessary for the llama-config
# package.
#
set(GGML_TRANSIENT_DEFINES)
get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
if (GGML_DIR_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_DIR_DEFINES})
endif()
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
if (GGML_TARGET_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES})
endif()
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)


#
# add utils, programs, demo..
#

add_subdirectory(src)




