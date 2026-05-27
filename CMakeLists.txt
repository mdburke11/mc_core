cmake_minimum_required(VERSION 3.18)
project(MonteCarloCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(HDF5 REQUIRED)

FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG        v2.10.1
)

set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(HighFive)

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

install(TARGETS mc_core
    EXPORT MonteCarloCoreTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)