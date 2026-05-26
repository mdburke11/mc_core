cmake_minimum_required(VERSION 3.18)
project(MonteCarloCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)

add_library(mc_core
    src/Params.cpp
    src/H5IO.cpp
)

target_include_directories(mc_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mc_core
    PUBLIC
        HighFive
        HDF5::HDF5
)

target_compile_features(mc_core PUBLIC cxx_std_17)

add_subdirectory(examples/ising2d)