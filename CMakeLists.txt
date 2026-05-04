cmake_minimum_required(VERSION 3.16)
project(dd_mpc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Third-party dependencies ─────────────────────────────────────────────────

find_package(Eigen3 REQUIRED)

# OSQP: expected as a subdirectory (git submodule).
# Set OSQP_BUILD_SHARED_LIBS=OFF so we always link statically.
set(OSQP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(osqp)

# ── MPC core library ─────────────────────────────────────────────────────────

add_library(mpc_core
    src/mpc_controller.cpp
    src/solver.cpp
    src/qp_builder.cpp
    src/reference.cpp
    src/path_smoother.cpp
    src/projection.cpp
    src/linearization.cpp
)

target_include_directories(mpc_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        osqp/include
        osqp/include/public
        ${CMAKE_BINARY_DIR}/osqp/include/public
)

target_link_libraries(mpc_core
    PUBLIC  Eigen3::Eigen
    PRIVATE osqpstatic
)

# ── Standalone simulation executable
add_executable(sim_test
    test/sim_main.cpp
    test/sim_utils.cpp
)
target_include_directories(sim_test PRIVATE test)
target_link_libraries(sim_test PRIVATE mpc_core)