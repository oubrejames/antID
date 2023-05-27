# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
message("CMAKE_INSTALL_PREFIX" ${CMAKE_INSTALL_PREFIX})

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

# target_compile_options(main PUBLIC -Wall -Wextra -Wpedantic)


# # Public include directories can be used by other targets that link against turtlelib
# # By adding include/ to the include path, this means that files in e.g., include/turtlelib
# # can be included with #include"turtlelib/file.hpp"
# target_include_directories(antID
#   PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)

# # warnings are your friend!
# # target_compile_options(black_fly_tools PUBLIC -Wall -Wextra -Wpedantic)
# target_compile_options(include PRIVATE -Wno-system-headers)


# install(DIRECTORY include/include DESTINATION include)
# install(TARGETS collect_data include EXPORT include-targets)

# install(EXPORT antID-targets
#   FILE antID-config.cmake
#   NAMESPACE antID::
#   DESTINATION lib/cmake/${PROJECT_NAME})