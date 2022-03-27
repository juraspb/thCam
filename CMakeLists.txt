cmake_minimum_required (VERSION 3.5)

project(typhoon)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(typhoon ${OpenCV_LIBS})

file(GLOB
        CPPS "source/*.cpp")

add_executable (${PROJECT_NAME} ${CPPS})

#add_executable (source/*.cpp)

#target_link_libraries (typhoon LINK_PUBLIC 
#                       ${OpenCV_LIBS})