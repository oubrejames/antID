# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(antID)

add_executable(collect_data src/collect_data.cpp)


# enable C++ 17
# target_compile_features(antID PUBLIC cxx_std_17) 

# Generate doxygen documentation
# find_package(Doxygen)
# set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
# doxygen_add_docs(doxygen include/ src/ README.md ALL)

find_package(black_fly_tools REQUIRED)
include_directories(${black_fly_tools_INCLUDE_DIRS})

find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})

find_package(turtlelib REQUIRED)
include_directories(${turtlelib_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# find_package(conio REQUIRED)
# include_directories(${conio_INCLUDE_DIRS})
target_link_libraries(collect_data black_fly_tools::black_fly_tools "${cpp_typesupport_target}")