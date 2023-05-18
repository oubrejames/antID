# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(black_fly_tools)

add_executable(main src/main.cpp)

# # create the library
# add_library(black_fly_tools)

# # The add_library function just added turtlelib as a "target"
# # A "target" is a name that CMake uses to refer to some type of output
# # In this case it is a library but it could also be an executable or some other items
# target_link_libraries(main black_fly_tools)

# enable C++ 17
target_compile_features(main PUBLIC cxx_std_17) 

# Generate doxygen documentation
# find_package(Doxygen)
# set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
# doxygen_add_docs(doxygen include/ src/ README.md ALL)

find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})
target_link_libraries(main ${Spinnaker_LIBRARIES})

find_package(OpenCV REQUIRED)
target_link_libraries(main ${OpenCV_LIBS})

target_compile_options(main PUBLIC -Wall -Wextra -Wpedantic)
