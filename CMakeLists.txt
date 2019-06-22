cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(SWCompiler C CXX)

# SWCompiler version
set(SWCOMPILER_VERSION_MAJOR 0)
set(SWCOMPILER_VERSION_MINOR 0)
set(SWCOMPILER_VERSION_ADDIT 0)
message(STATUS "This is SWCompiler ${SWCOMPILER_VERSION_MAJOR}.${SWCOMPILER_VERSION_MINOR}.${SWCOMPILER_VERSION_ADDIT}!")

# Build Type
if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_definitions(-DDEBUG)
endif()

if(LEVELDEBUG)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DLEVELDEBUG=${LEVELDEBUG})
  else()
    add_definitions(-DLEVELDEBUG=100)
  endif()
endif ()

option(ONNX_IMPORT_ENABLE "Enable ONNX importer" FALSE)

set(SW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SW_BINARY_DIR})

# Cmake Modules
include(cmake/Util.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Options
# Architecture level
# Optimised level


# Headers
include_directories(${PROJECT_SOURCE_DIR}/src)

# Generic compilation options
# Flags
set (COMMON_FLAGS "-Wall -Wno-uninitialized -Wno-deprecated-declarations -Wno-unknown-pragmas -Wsign-compare ${ADDITIONAL_FLAGS}")
set (CMAKE_C_FLAGS "${COMMON_FLAGS} -fno-strict-aliasing")
set (CMAKE_CXX_FLAGS "${COMMON_FLAGS} -std=c++0x")
# Definitions

## copy utils/* instead to SW_BINARY_DIR/utils
# set(CODEGEN_HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/utils/image.h)
# message(STATUS "Copy File ${CODEGEN_HDR} to ${SW_BINARY_DIR}")
# add_custom_command(
#   OUTPUT ${SW_BINARY_DIR}/image.h
#   COMMAND ${CMAKE_COMMAND}
#   -E copy_if_different ${CODEGEN_HDR} ${SW_BINARY_DIR}/image.h
#   DEPENDS ${CODEGEN_HDR})
# set_property(SOURCE ${PROJECT_SOURCE_DIR}/src/codegen/Codegen.cpp APPEND PROPERTY OBJECT_DEPENDS ${SW_BINARY_DIR}/image.h)

add_subdirectory(src)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

# protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

PROTOBUF_GENERATE_CPP(CAFFE2_SRCS CAFFE2_HDRS src/importer/caffe2.proto)
PROTOBUF_GENERATE_CPP(ONNX_SRCS ONNX_HDRS src/importer/onnx-ml.proto)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()
message(STATUS "EXTERNAL_PROJECTS_ROOT ${EXTERNAL_PROJECTS_ROOT}")
if (ONNX_IMPORT_ENABLE)
    include(cmake/external_onnx.cmake)
endif()

# Source
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp)
# add_library(graphcore STATIC ${SOURCES})
add_library(graphcore
			STATIC
			${SOURCES}
      ${CAFFE2_SRCS}
      ${ONNX_SRCS})
      # ${CMAKE_CURRENT_BINARY_DIR}/caffe2.pb.h)
target_compile_definitions(graphcore
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
                             
target_link_libraries(graphcore PUBLIC 
                        ${PROTOBUF_LIBRARY}
                        Threads::Threads)
#-----------------------------------------------------------------------------------------------
# ONNX support tmp.
#-----------------------------------------------------------------------------------------------
if (ONNX_IMPORT_ENABLE)
    set(ONNX_OPSET_VERSION 9 CACHE INTERNAL "Supported version of ONNX operator set")

    add_dependencies(graphcore onnx::libonnx onnx::libonnx_proto)

    target_include_directories(graphcore  SYSTEM PRIVATE 
        ${ONNX_INCLUDE_DIR} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIR})
    target_link_libraries(graphcore PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY})

    target_compile_definitions(graphcore PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})
endif()
# Libraries are compiled
link_libraries(graphcore)


# now build binaries
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(test)
# file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/*.cpp)
# foreach(TESTSRC ${TEST_SOURCES})
#    get_filename_component(TESTNAME ${TESTSRC} NAME_WE) # file name without directories or longest extension
#    add_executable(${TESTNAME} ${TESTSRC})
#    install(TARGETS ${TESTNAME} RUNTIME DESTINATION bin)
# endforeach()
