cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

message("CMAKE_INSTALL_PREFIX" ${CMAKE_INSTALL_PREFIX})

# set the project name
project(black_fly_tools)

add_executable(stream src/stream.cpp)
add_executable(record src/record.cpp)

# create the library
add_library(black_fly_tools src/black_fly_lib.cpp)

# Link the executables to the library
target_link_libraries(stream black_fly_tools)
target_link_libraries(record black_fly_tools)

# enable C++ 17
target_compile_features(black_fly_tools PUBLIC cxx_std_17) 

# Generate doxygen documentation
find_package(Doxygen)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
doxygen_add_docs(doxygen include/ src/ README.md ALL)

# Add dependencies
find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})

target_link_libraries(black_fly_tools ${Spinnaker_LIBRARIES})
target_link_directories(black_fly_tools PUBLIC ${Spinnaker_LIBRARY_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_directories(black_fly_tools PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(black_fly_tools ${OpenCV_LIBS})

target_include_directories(black_fly_tools
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Compiler warnings
target_compile_options(black_fly_tools PUBLIC -Wall -Wextra -Wpedantic)

install(DIRECTORY include/black_fly_tools DESTINATION include)
install(TARGETS record stream black_fly_tools EXPORT black_fly_tools-targets)

install(EXPORT black_fly_tools-targets
  FILE black_fly_tools-config.cmake
  NAMESPACE black_fly_tools::
  DESTINATION lib/cmake/${PROJECT_NAME})