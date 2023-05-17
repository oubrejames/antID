# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(antid)

add_executable(${PROJECT_NAME} src/main.cpp)

