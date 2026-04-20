cmake_minimum_required(VERSION 3.16)
project(dd_mpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Dependencies
# -------------------------

find_package(Eigen3 REQUIRED)

# Add OSQP (assuming cloned locally)
add_subdirectory(osqp)

# -------------------------
# MPC Library
# -------------------------

file(GLOB MPC_HEADERS include/mpc/*.hpp)
file(GLOB MPC_SOURCES src/*.cpp)

add_library(mpc_core
    ${MPC_HEADERS}
    ${MPC_SOURCES}
)

target_include_directories(mpc_core
    PUBLIC
        include
        osqp/include
        osqp/include/public
        ${CMAKE_BINARY_DIR}/osqp/include/public
)

target_link_libraries(mpc_core
    PUBLIC Eigen3::Eigen
    PRIVATE osqpstatic
)

# -------------------------
# Test Executable
# -------------------------

add_executable(sim_test
    test/sim_main.cpp
)

target_link_libraries(sim_test
    mpc_core
)
