cmake_minimum_required(VERSION 3.5.1)
project(java_torch_lib)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_PREFIX_PATH /home/nazar/libtorch)
message(STATUS "torch lib path is ${CMAKE_PREFIX_PATH}")
# link_directories(/usr/local/cuda-9.0/lib64)

find_library(curand CURAND)

find_package(Torch REQUIRED)

message (STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message (STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")

include_directories(${TORCH_INCLUDE_DIRS})

# include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include)
# include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include/linux)

find_package(JNI)

if (${JNI_FOUND})
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

include_directories(${JNI_INCLUDE_DIRS})

set (LIB_NAME ${PROJECT_NAME}${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED jnijavacpp.cpp java_torch_lib.cpp)
target_link_libraries(${LIB_NAME}  "${TORCH_LIBRARIES}")
target_link_libraries(${LIB_NAME} ${JNI_LIBRARIES})
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 11)


install(TARGETS ${LIB_NAME} LIBRARY DESTINATION .)