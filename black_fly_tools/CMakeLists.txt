# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(black_fly_tools)

add_executable(stream src/stream.cpp)
add_executable(record src/record.cpp)

# create the library
add_library(black_fly_tools src/black_fly_lib.cpp)

# The add_library function just added turtlelib as a "target"
# A "target" is a name that CMake uses to refer to some type of output
# In this case it is a library but it could also be an executable or some other items
target_link_libraries(stream black_fly_tools)
target_link_libraries(record black_fly_tools)

# enable C++ 17
target_compile_features(black_fly_tools PUBLIC cxx_std_17) 

# Generate doxygen documentation
# find_package(Doxygen)
# set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
# doxygen_add_docs(doxygen include/ src/ README.md ALL)

find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})
target_link_libraries(black_fly_tools ${Spinnaker_LIBRARIES})

find_package(OpenCV REQUIRED)
target_link_libraries(black_fly_tools ${OpenCV_LIBS})

# target_compile_options(main PUBLIC -Wall -Wextra -Wpedantic)


# Public include directories can be used by other targets that link against turtlelib
# By adding include/ to the include path, this means that files in e.g., include/turtlelib
# can be included with #include"turtlelib/file.hpp"
target_include_directories(black_fly_tools
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# warnings are your friend!
# target_compile_options(black_fly_tools PUBLIC -Wall -Wextra -Wpedantic)
target_compile_options(black_fly_tools PRIVATE -Wno-system-headers)


install(DIRECTORY include/black_fly_tools DESTINATION include)
install(TARGETS stream record black_fly_tools EXPORT black_fly_tools-targets)

install(EXPORT black_fly_tools-targets
  FILE black_fly_tools-config.cmake
  NAMESPACE black_fly_tools::
  DESTINATION lib/cmake/${PROJECT_NAME})