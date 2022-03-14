cmake_minimum_required(VERSION 3.10)

project(vpi_stereo_disparity)

find_package(vpi 1.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(DEBUG 1)

if(DEBUG)
  set(OUTPUT_DIR Debug)
else()
  set(OUTPUT_DIR Release)
endif()

include_directories(include) 
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${OUTPUT_DIR})


add_executable(${PROJECT_NAME} main.cpp Capture.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} vpi)
