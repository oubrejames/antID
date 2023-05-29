# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
# set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory")
message("CMAKE_INSTALL_PREFIX" ${CMAKE_INSTALL_PREFIX})
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
target_link_directories(black_fly_tools PUBLIC ${Spinnaker_LIBRARY_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_directories(black_fly_tools PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(black_fly_tools ${OpenCV_LIBS})

# target_compile_options(main PUBLIC -Wall -Wextra -Wpedantic)


# Public include directories can be used by other targets that link against turtlelib
# By adding include/ to the include path, this means that files in e.g., include/turtlelib
# can be included with #include"turtlelib/file.hpp"
# target_include_directories(black_fly_tools
#   PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)

# # warnings are your friend!
# target_compile_options(black_fly_tools PUBLIC -Wall -Wextra -Wpedantic)

target_include_directories(black_fly_tools
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)



# warnings are your friend!
target_compile_options(black_fly_tools PUBLIC -Wall -Wextra -Wpedantic)

install(DIRECTORY include/black_fly_tools DESTINATION include)
install(TARGETS record stream black_fly_tools EXPORT black_fly_tools-targets)

install(EXPORT black_fly_tools-targets
  FILE black_fly_tools-config.cmake
  NAMESPACE black_fly_tools::
  DESTINATION lib/cmake/${PROJECT_NAME})

##########################################################################
# # target_compile_options(black_fly_tools PRIVATE -Wno-system-headers)


# # install(DIRECTORY include/black_fly_tools DESTINATION include)
# # install(TARGETS stream record black_fly_tools EXPORT black_fly_tools-targets)

# # install(EXPORT black_fly_tools-targets
# #   FILE black_fly_tools-config.cmake
# #   NAMESPACE black_fly_tools::
# #   DESTINATION lib/cmake/${PROJECT_NAME})

# ###################
# # The install(TARGETS ...) command specifies the installation destinations for the targets (stream, 
# # record, and black_fly_tools). The executables will be installed to the bin directory, and the 
# # library files (lib) will be installed to the lib directory.
# install(TARGETS black_fly_tools
#         EXPORT black_fly_tools-targets
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
# )

# # The install(DIRECTORY ...) command is used to install the header files located in the 
# # include/black_fly_tools directory to the include directory.
# install(DIRECTORY include/black_fly_tools
#         DESTINATION include
# )
# #  install(EXPORT ...) command generates the export file black_fly_tools-config.cmake and installs 
# # it to the lib/cmake/${PROJECT_NAME} directory. This file allows other projects to import and use 
# # the black_fly_tools target as a dependency.
# install(EXPORT black_fly_tools-targets
#         FILE black_fly_tools-config.cmake
#         NAMESPACE black_fly_tools::
#         DESTINATION lib/cmake/${PROJECT_NAME}
# )