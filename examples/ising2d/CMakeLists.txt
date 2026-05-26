add_executable(ising2d
    main.cpp
    Ising2D.cpp
)

target_include_directories(ising2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ising2d PRIVATE mc_core)