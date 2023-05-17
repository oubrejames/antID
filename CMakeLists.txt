# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(antid)

add_executable(${PROJECT_NAME} src/main.cpp)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR
    "No in source builds allowed. Create a separate build directory.
    SOURCE_DIR=${CMAKE_SOURCE_DIR}  BINARY_DIR=${CMAKE_BINARY_DIR} ")
endif()