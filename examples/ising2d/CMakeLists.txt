cmake_minimum_required(VERSION 3.18)
project(Ising2DExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MC_CORE_ROOT "$ENV{HOME}/codes/mc_core/install" CACHE PATH "mc_core install prefix")

message(STATUS "MC_CORE_ROOT = ${MC_CORE_ROOT}")

set(MC_CORE_INCLUDE_DIR "${MC_CORE_ROOT}/include")
set(MC_CORE_LIBRARY "${MC_CORE_ROOT}/lib/libmc_core.a")

message(STATUS "MC_CORE_INCLUDE_DIR = ${MC_CORE_INCLUDE_DIR}")
message(STATUS "MC_CORE_LIBRARY = ${MC_CORE_LIBRARY}")

if(NOT EXISTS "${MC_CORE_INCLUDE_DIR}/mc/AnnealingRunner.hpp")
    message(FATAL_ERROR
        "Could not find AnnealingRunner.hpp at "
        "${MC_CORE_INCLUDE_DIR}/mc/AnnealingRunner.hpp"
    )
endif()

if(NOT EXISTS "${MC_CORE_LIBRARY}")
    message(FATAL_ERROR
        "Could not find libmc_core.a at ${MC_CORE_LIBRARY}"
    )
endif()

find_package(HDF5 REQUIRED)

add_executable(ising2d
    src/main.cpp
    src/Ising2D.cpp
)

target_include_directories(ising2d PRIVATE
    src
    "${MC_CORE_INCLUDE_DIR}"
)

target_link_libraries(ising2d PRIVATE
    "${MC_CORE_LIBRARY}"
    HDF5::HDF5
)
