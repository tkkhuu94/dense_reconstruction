cmake_minimum_required(VERSION 3.0)
project(dense_reconstruction)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(
    dense_reconstruction 
    src/main.cpp
    src/RGBDCamera.cpp
)

target_link_libraries(
    dense_reconstruction 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
)
