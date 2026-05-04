cmake_minimum_required(VERSION 3.16)
project(dd_mpc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Detect build environment ─────────────────────────────────────────────────
# The same CMakeLists.txt serves two modes:
#   - Plain CMake  : builds mpc_core library + sim_test executable only.
#   - ament_cmake  : additionally builds the mpc_node ROS 2 node.
find_package(ament_cmake QUIET)

# ── Third-party dependencies ─────────────────────────────────────────────────

find_package(Eigen3 REQUIRED)

# OSQP: expected as a subdirectory (git submodule).
# Set OSQP_BUILD_SHARED_LIBS=OFF so we always link statically.
set(OSQP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(osqp)

# ── MPC core library ─────────────────────────────────────────────────────────
# Explicit source list instead of GLOB so that mpc_node.cpp is never
# accidentally included in the library.

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

# ── Standalone simulation executable ─────────────────────────────────────────
# sim_utils.cpp contains all testing utilities (noise, stuck detection,
# logging) and is compiled only into the sim executable — not into mpc_core —
# so the core library stays free of test-only dependencies.
add_executable(sim_test
    test/sim_main.cpp
    test/sim_utils.cpp
)
target_include_directories(sim_test PRIVATE test)
target_link_libraries(sim_test PRIVATE mpc_core)

# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 section (skipped entirely when ament_cmake is not present)
# ─────────────────────────────────────────────────────────────────────────────
if(ament_cmake_FOUND)
    message(STATUS "[dd_mpc] ament_cmake found — building ROS 2 node.")

    find_package(rclcpp            REQUIRED)
    find_package(geometry_msgs     REQUIRED)
    find_package(nav_msgs          REQUIRED)
    find_package(std_msgs          REQUIRED)
    find_package(std_srvs          REQUIRED)
    find_package(visualization_msgs REQUIRED)
    find_package(diagnostic_msgs   REQUIRED)
    find_package(tf2               REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)

    # ── ROS 2 node executable ─────────────────────────────────────────
    add_executable(mpc_node src/mpc_node.cpp)

    target_link_libraries(mpc_node mpc_core)

    ament_target_dependencies(mpc_node
        rclcpp
        geometry_msgs
        nav_msgs
        std_msgs
        std_srvs
        visualization_msgs
        diagnostic_msgs
        tf2
        tf2_geometry_msgs
    )

    # ── Install ───────────────────────────────────────────────────────
    install(
        TARGETS mpc_core mpc_node sim_test
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib/${PROJECT_NAME}
    )

    install(
        DIRECTORY include/
        DESTINATION include
    )

    install(
        DIRECTORY launch/
        DESTINATION share/${PROJECT_NAME}/launch
    )

    install(
        DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}/config
        OPTIONAL   # config/ may not exist yet
    )

    ament_package()
else()
    message(STATUS "[dd_mpc] ament_cmake NOT found — building standalone sim only.")
endif()