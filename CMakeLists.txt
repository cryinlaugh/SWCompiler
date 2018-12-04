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

# Cmake Modules
include(cmake/Util.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Options
# Architecture level
# Optimised level


# Headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Generic compilation options 
# Flags
set (COMMON_FLAGS "-Wall -Wno-uninitialized -Wno-deprecated-declarations -Wno-unknown-pragmas -Wsign-compare ${ADDITIONAL_FLAGS}")
set (CMAKE_C_FLAGS "${COMMON_FLAGS} -fno-strict-aliasing")
set (CMAKE_CXX_FLAGS "${COMMON_FLAGS} -std=c++0x")
# Definitions


# Source 
file(GLOB_RECURSE SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(graphcore STATIC ${SOURCES})


# Libraries are compiled. Now compile the programs
link_libraries(graphcore)

file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/testsrc/*.cpp)
foreach(TESTSRC ${TEST_SOURCES})
    get_filename_component(TESTNAME ${TESTSRC} NAME_WE) # file name without directories or longest extension
    add_executable(${TESTNAME} ${TESTSRC})
    install(TARGETS ${TESTNAME} RUNTIME DESTINATION bin)
endforeach()
