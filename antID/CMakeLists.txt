# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(antID)

# add_executable(collect_data src/collect_data.cpp)
# add_executable(edge_tune examples/edge_feature_tuning.cpp)
add_executable(video_to_images src/video_to_images.cpp)


# enable C++ 17
# target_compile_features(antID PUBLIC cxx_std_17) 

# Generate doxygen documentation
# find_package(Doxygen)
# set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
# doxygen_add_docs(doxygen include/ src/ README.md ALL)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package(conio REQUIRED)
# include_directories(${conio_INCLUDE_DIRS})
target_link_libraries(collect_data ${GTK_LIBRARIES} ${OpenCV_LIBRARIES}  "${cpp_typesupport_target}")
target_link_libraries(edge_tune ${GTK_LIBRARIES} ${OpenCV_LIBRARIES}  "${cpp_typesupport_target}")
target_link_libraries(video_to_images ${GTK_LIBRARIES} ${OpenCV_LIBRARIES} "${cpp_typesupport_target}")