"""
launch/mpc.launch.py  —  Launch the MPC node + optional foxglove_bridge

Usage
-----
# Basic (no Foxglove bridge):
  ros2 launch dd_mpc mpc.launch.py

# With Foxglove bridge on the default port 8765:
  ros2 launch dd_mpc mpc.launch.py foxglove:=true

# Override MPC parameters inline:
  ros2 launch dd_mpc mpc.launch.py N:=25 v_ref:=0.5 d_hard:=0.06

# Load parameters from a YAML file:
  ros2 launch dd_mpc mpc.launch.py params_file:=/path/to/mpc_params.yaml

Foxglove
--------
After launch, open https://app.foxglove.dev (or the desktop app) and connect
to ws://localhost:8765.  Useful panels:
  - 3D              → subscribe to /mpc/full_path, /mpc/ref_path,
                        /mpc/pred_path, /mpc/debug/corridor,
                        /mpc/debug/seg_normals
  - Plot            → /mpc/debug/cte, /mpc/debug/d_hard_eff,
                        /mpc/debug/v_scale, /mpc/debug/q_theta_eff
  - Diagnostics     → /mpc/debug/diagnostics
  - Raw Messages    → /mpc/debug/v_profile
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Declare arguments ─────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("foxglove",    default_value="false",
                              description="Launch foxglove_bridge"),
        DeclareLaunchArgument("foxglove_port", default_value="8765",
                              description="WebSocket port for foxglove_bridge"),
        DeclareLaunchArgument("params_file", default_value="",
                              description="Path to a YAML params file (optional)"),
        DeclareLaunchArgument("map_frame",   default_value="map",
                              description="TF frame used for all published topics"),

        # ── MPC tuning knobs (all have sensible defaults) ────────────
        DeclareLaunchArgument("N",                    default_value="20"),
        DeclareLaunchArgument("dt",                   default_value="0.05"),
        DeclareLaunchArgument("v_ref",                default_value="0.3"),
        DeclareLaunchArgument("v_max",                default_value="0.5"),
        DeclareLaunchArgument("v_min",                default_value="-0.1"),
        DeclareLaunchArgument("omega_max",            default_value="1.5"),
        DeclareLaunchArgument("a_max",                default_value="1.0"),
        DeclareLaunchArgument("alpha_max",            default_value="2.0"),
        DeclareLaunchArgument("d_hard",               default_value="0.05"),
        DeclareLaunchArgument("adaptive_corridor_scale", default_value="3.0"),
        DeclareLaunchArgument("funnel_decay_tau",     default_value="10.0"),
        DeclareLaunchArgument("w_slack",              default_value="5000.0"),
        DeclareLaunchArgument("Q_xy",                 default_value="60.0"),
        DeclareLaunchArgument("Q_theta",              default_value="10.0"),
        DeclareLaunchArgument("Q_xy_terminal",        default_value="60.0"),
        DeclareLaunchArgument("Q_theta_terminal",     default_value="100.0"),
        DeclareLaunchArgument("R_v",                  default_value="5.0"),
        DeclareLaunchArgument("R_omega",              default_value="5.0"),
        DeclareLaunchArgument("R_rate_v",             default_value="2.0"),
        DeclareLaunchArgument("R_rate_omega",         default_value="5.0"),
        DeclareLaunchArgument("heading_scale_k",      default_value="1.0"),
        DeclareLaunchArgument("blend_alpha",          default_value="0.7"),
        DeclareLaunchArgument("goal_threshold",       default_value="0.3"),
        DeclareLaunchArgument("d_deadband",           default_value="0.02"),
        DeclareLaunchArgument("v_error_gain",         default_value="3.0"),
        DeclareLaunchArgument("v_min_scale",          default_value="0.2"),
        DeclareLaunchArgument("path_reset_threshold", default_value="0.5"),
        DeclareLaunchArgument("fallback_decay",       default_value="0.8"),
    ]

    # ── MPC node ──────────────────────────────────────────────────────
    def make_mpc_node(context):
        params_file = LaunchConfiguration("params_file").perform(context)

        # Inline parameters (always applied; YAML file takes lower priority)
        inline_params = {
            "map_frame":              LaunchConfiguration("map_frame"),
            "N":                      LaunchConfiguration("N"),
            "dt":                     LaunchConfiguration("dt"),
            "v_ref":                  LaunchConfiguration("v_ref"),
            "v_max":                  LaunchConfiguration("v_max"),
            "v_min":                  LaunchConfiguration("v_min"),
            "omega_max":              LaunchConfiguration("omega_max"),
            "a_max":                  LaunchConfiguration("a_max"),
            "alpha_max":              LaunchConfiguration("alpha_max"),
            "d_hard":                 LaunchConfiguration("d_hard"),
            "adaptive_corridor_scale": LaunchConfiguration("adaptive_corridor_scale"),
            "funnel_decay_tau":       LaunchConfiguration("funnel_decay_tau"),
            "w_slack":                LaunchConfiguration("w_slack"),
            "Q_xy":                   LaunchConfiguration("Q_xy"),
            "Q_theta":                LaunchConfiguration("Q_theta"),
            "Q_xy_terminal":          LaunchConfiguration("Q_xy_terminal"),
            "Q_theta_terminal":       LaunchConfiguration("Q_theta_terminal"),
            "R_v":                    LaunchConfiguration("R_v"),
            "R_omega":                LaunchConfiguration("R_omega"),
            "R_rate_v":               LaunchConfiguration("R_rate_v"),
            "R_rate_omega":           LaunchConfiguration("R_rate_omega"),
            "heading_scale_k":        LaunchConfiguration("heading_scale_k"),
            "blend_alpha":            LaunchConfiguration("blend_alpha"),
            "goal_threshold":         LaunchConfiguration("goal_threshold"),
            "d_deadband":             LaunchConfiguration("d_deadband"),
            "v_error_gain":           LaunchConfiguration("v_error_gain"),
            "v_min_scale":            LaunchConfiguration("v_min_scale"),
            "path_reset_threshold":   LaunchConfiguration("path_reset_threshold"),
            "fallback_decay":         LaunchConfiguration("fallback_decay"),
        }

        node_params = [inline_params]
        if params_file:
            node_params.insert(0, params_file)  # YAML has higher priority

        return [
            Node(
                package="dd_mpc",
                executable="mpc_node",
                name="mpc_node",
                output="screen",
                emulate_tty=True,
                parameters=node_params,
                remappings=[
                    # Remap if your robot uses different topic names:
                    # ("odom",     "/robot/odom"),
                    # ("path",     "/planner/path"),
                    # ("cmd_vel",  "/robot/cmd_vel"),
                ],
            )
        ]

    # ── Optional Foxglove bridge ──────────────────────────────────────
    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{
            "port": LaunchConfiguration("foxglove_port"),
            # Increase send buffer for high-frequency marker arrays
            "send_buffer_limit": 10_000_000,
        }],
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    return LaunchDescription(args + [OpaqueFunction(function=make_mpc_node),
                                     foxglove_bridge])