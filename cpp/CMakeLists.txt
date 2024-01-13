cmake_minimum_required(VERSION 3.15)

project(sennet_cpp CXX)

find_package(CLI11 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(mio CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 23)

include_directories(src/library)
add_subdirectory(src)
