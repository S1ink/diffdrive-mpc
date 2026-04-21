// =============================================================================
// mpc_node.cpp  —  Self-contained MPC simulator ROS 2 node
//
// The node owns the plant model and steps it internally on a wall timer.
// No /odom subscription is required.  Publishing to /path at any time
// overrides the built-in scenario path.
//
// Scenario paths (ROS parameter `scenario`, set at startup):
//   0  Straight 0→10 m, small lateral offset  (default; path updates at t=7.5s)
//   1  Sharp L-turn
//   2  Figure-8
//   3  Straight with a badly-initialised pose  (tests recovery)
//
// Services:
//   /mpc/pause      std_srvs/SetBool    true = pause, false = resume
//   /mpc/restart    std_srvs/Trigger    reset plant + controller to scenario IC
//   /mpc/reset      std_srvs/Trigger    controller hard-reset only (plant runs on)
//
// Published topics:
//   /cmd_vel                        geometry_msgs/Twist
//   /mpc/full_path                  nav_msgs/Path
//   /mpc/ref_path                   nav_msgs/Path
//   /mpc/pred_path                  nav_msgs/Path
//   /mpc/debug/robot_pose           geometry_msgs/PoseStamped
//   /mpc/debug/proj_point           geometry_msgs/PointStamped
//   /mpc/debug/cte                  std_msgs/Float64
//   /mpc/debug/d_hard_eff           std_msgs/Float64
//   /mpc/debug/v_scale              std_msgs/Float64
//   /mpc/debug/q_theta_eff          std_msgs/Float64
//   /mpc/debug/solver_ok            std_msgs/Bool
//   /mpc/debug/near_goal            std_msgs/Bool
//   /mpc/debug/v_profile            std_msgs/Float64MultiArray
//   /mpc/debug/corridor             visualization_msgs/MarkerArray
//   /mpc/debug/seg_normals          visualization_msgs/MarkerArray
//   /mpc/debug/diagnostics          diagnostic_msgs/DiagnosticStatus
// =============================================================================

#include <rclcpp/rclcpp.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cmath>

#include "mpc/mpc_controller.hpp"
#include "mpc/params.hpp"
#include "mpc/path.hpp"
#include "mpc/types.hpp"

using namespace std::chrono_literals;

// ── Helpers ───────────────────────────────────────────────────────────────────

static geometry_msgs::msg::Quaternion yawToQuat(double yaw)
{
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    return tf2::toMsg(q);
}

// ── Path factories ────────────────────────────────────────────────────────────

static mpc::Path
    makeStraightPath(double x0, double x1, double y = 0.0, int pts = 20)
{
    mpc::Path p;
    for (int i = 0; i <= pts; ++i)
    {
        const double t = static_cast<double>(i) / pts;
        p.pts.push_back({Eigen::Vector2d(x0 + t * (x1 - x0), y)});
    }
    return p;
}

static mpc::Path makeSharpLPath()
{
    mpc::Path p;
    p.pts.push_back({Eigen::Vector2d(0, 0)});
    p.pts.push_back({Eigen::Vector2d(1, 0)});
    p.pts.push_back({Eigen::Vector2d(2, 0)});
    p.pts.push_back({Eigen::Vector2d(2, 0.1)});
    p.pts.push_back({Eigen::Vector2d(2, 1)});
    p.pts.push_back({Eigen::Vector2d(2, 2)});
    return p;
}

static mpc::Path makeFigure8Path(int n = 100, double scale = 3.0)
{
    mpc::Path p;
    for (int i = 0; i <= n; ++i)
    {
        const double t = 2.0 * M_PI * i / n;
        p.pts.push_back({Eigen::Vector2d(
            scale * std::sin(t),
            scale * std::sin(t) * std::cos(t))});
    }
    return p;
}

// ── Node ──────────────────────────────────────────────────────────────────────

class MPCNode : public rclcpp::Node
{
public:
    explicit MPCNode(
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) :
        Node("mpc_node", options)
    {
        declareAndLoadParams();

        controller_ = std::make_unique<mpc::MPCController>(params_);
        initScenario();  // sets path_ and x_sim_

        // ── External path override ────────────────────────────────────
        // Publishing to /path replaces the built-in scenario path and
        // resets the controller so projection restarts cleanly.
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "path",
            rclcpp::QoS(10).transient_local(),
            [this](nav_msgs::msg::Path::ConstSharedPtr msg)
            {
                mpc::Path p;
                p.pts.reserve(msg->poses.size());
                for (const auto& ps : msg->poses)
                {
                    p.pts.push_back({Eigen::Vector2d(
                        ps.pose.position.x,
                        ps.pose.position.y)});
                }

                if (!p.valid())
                {
                    RCLCPP_WARN(
                        get_logger(),
                        "Received path with < 2 points — ignoring.");
                    return;
                }
                path_ = std::move(p);
                controller_->reset();
                publishFullPath(now());
                RCLCPP_INFO(
                    get_logger(),
                    "External path received (%zu pts) — controller reset.",
                    path_.pts.size());
            });

        // ── Publishers ────────────────────────────────────────────────
        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        ref_pub_ = create_publisher<nav_msgs::msg::Path>("mpc/ref_path", 10);
        pred_pub_ = create_publisher<nav_msgs::msg::Path>("mpc/pred_path", 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>("mpc/full_path", 10);
        cte_pub_ =
            create_publisher<std_msgs::msg::Float64>("mpc/debug/cte", 10);
        d_hard_pub_ = create_publisher<std_msgs::msg::Float64>(
            "mpc/debug/d_hard_eff",
            10);
        v_scale_pub_ =
            create_publisher<std_msgs::msg::Float64>("mpc/debug/v_scale", 10);
        q_theta_pub_ = create_publisher<std_msgs::msg::Float64>(
            "mpc/debug/q_theta_eff",
            10);
        solver_ok_pub_ =
            create_publisher<std_msgs::msg::Bool>("mpc/debug/solver_ok", 10);
        near_goal_pub_ =
            create_publisher<std_msgs::msg::Bool>("mpc/debug/near_goal", 10);
        proj_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "mpc/debug/proj_point",
            10);
        v_profile_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
            "mpc/debug/v_profile",
            10);
        robot_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "mpc/debug/robot_pose",
            10);
        corridor_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "mpc/debug/corridor",
            10);
        normals_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "mpc/debug/seg_normals",
            10);
        diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticStatus>(
            "mpc/debug/diagnostics",
            10);

        // ── Services ──────────────────────────────────────────────────

        // /mpc/pause  (SetBool: true=pause, false=resume)
        pause_srv_ = create_service<std_srvs::srv::SetBool>(
            "mpc/pause",
            [this](
                std_srvs::srv::SetBool::Request::ConstSharedPtr req,
                std_srvs::srv::SetBool::Response::SharedPtr res)
            {
                paused_ = req->data;
                res->success = true;
                res->message =
                    paused_ ? "Simulation paused." : "Simulation resumed.";
                RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
            });

        // /mpc/restart  — reset plant + controller back to scenario initial conditions
        restart_srv_ = create_service<std_srvs::srv::Trigger>(
            "mpc/restart",
            [this](
                std_srvs::srv::Trigger::Request::ConstSharedPtr,
                std_srvs::srv::Trigger::Response::SharedPtr res)
            {
                initScenario();
                controller_->reset();
                step_count_ = 0;
                sim_time_ = 0.0;
                paused_ = false;
                publishFullPath(now());
                RCLCPP_INFO(
                    get_logger(),
                    "Simulation restarted (scenario %d).",
                    scenario_);
                res->success = true;
                res->message = "Restarted OK";
            });

        // /mpc/reset  — controller hard-reset only; plant keeps moving
        ctrl_reset_srv_ = create_service<std_srvs::srv::Trigger>(
            "mpc/reset",
            [this](
                std_srvs::srv::Trigger::Request::ConstSharedPtr,
                std_srvs::srv::Trigger::Response::SharedPtr res)
            {
                controller_->reset();
                RCLCPP_INFO(
                    get_logger(),
                    "Controller reset (plant continues at t=%.2fs).",
                    sim_time_);
                res->success = true;
                res->message = "Controller reset OK";
            });

        // ── Simulation timer ──────────────────────────────────────────
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(params_.dt)),
            [this]() { timerCallback(); });

        // Give Foxglove the initial path geometry immediately
        publishFullPath(now());

        RCLCPP_INFO(
            get_logger(),
            "MPC sim node ready — scenario=%d  dt=%.3fs (%.1f Hz)  frame='%s'\n"
            "  Services: /mpc/pause (SetBool)  /mpc/restart  /mpc/reset\n"
            "  Override path by publishing nav_msgs/Path to /path",
            scenario_,
            params_.dt,
            1.0 / params_.dt,
            frame_id_.c_str());
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Timer callback — one simulation step
    // ─────────────────────────────────────────────────────────────────────────

    void timerCallback()
    {
        if (paused_)
        {
            return;
        }

        // Scenario 0: mid-run path swap at step 150 (t ≈ 7.5 s)
        if (scenario_ == 0 && step_count_ == 150)
        {
            path_ = makeStraightPath(3.0, 5.0, 0.3);
            const rclcpp::Time t = now();
            publishFullPath(t);
            RCLCPP_INFO(
                get_logger(),
                "Path updated at t=%.2fs (step %d).",
                sim_time_,
                step_count_);
        }

        const rclcpp::Time stamp = now();

        // Run MPC on the current simulated state
        const mpc::Control u = controller_->update(x_sim_, path_);
        const mpc::DebugInfo& dbg = controller_->debugInfo();

        // Step nonlinear plant
        x_sim_ = plantStep(x_sim_, u, params_.dt);

        publishTwist(u);
        publishRobotPose(stamp);
        publishAllDebug(dbg, stamp);

        sim_time_ += params_.dt;
        ++step_count_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Nonlinear diff-drive plant  (ground truth)
    // ─────────────────────────────────────────────────────────────────────────

    static mpc::State
        plantStep(const mpc::State& x, const mpc::Control& u, double dt)
    {
        return {
            x.x + u.v * std::cos(x.theta) * dt,
            x.y + u.v * std::sin(x.theta) * dt,
            x.theta + u.omega * dt};
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scenario initialisation  (sets path_ and x_sim_)
    // ─────────────────────────────────────────────────────────────────────────

    void initScenario()
    {
        switch (scenario_)
        {
            case 1:
                path_ = makeSharpLPath();
                x_sim_ = {0.0, 0.25, 0.1};
                break;
            case 2:
                path_ = makeFigure8Path();
                x_sim_ = {0.0, 0.5, 0.0};
                break;
            case 3:
                path_ = makeStraightPath(0.0, 10.0);
                x_sim_ = {-2.0, 3.0, M_PI};  // bad initial pose
                break;
            default:  // 0
                path_ = makeStraightPath(0.0, 10.0);
                x_sim_ = {0.0, 0.25, 0.1};
                break;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Parameter loading
    // ─────────────────────────────────────────────────────────────────────────

    void declareAndLoadParams()
    {
        auto dbl = [this](const std::string& n, double def) -> double
        {
            declare_parameter(n, def);
            return get_parameter(n).as_double();
        };
        auto intP = [this](const std::string& n, int def) -> int
        {
            declare_parameter(n, def);
            return get_parameter(n).as_int();
        };

        scenario_ = intP("scenario", 0);

        params_.N = intP("N", 20);
        params_.dt = dbl("dt", 0.05);
        params_.v_max = dbl("v_max", 0.5);
        params_.v_min = dbl("v_min", -0.1);
        params_.omega_max = dbl("omega_max", 1.5);
        params_.a_max = dbl("a_max", 1.0);
        params_.alpha_max = dbl("alpha_max", 2.0);
        params_.v_ref = dbl("v_ref", 0.3);

        params_.d_hard = dbl("d_hard", 0.05);
        params_.w_slack = dbl("w_slack", 5000.0);
        params_.adaptive_corridor_scale = dbl("adaptive_corridor_scale", 3.0);
        params_.funnel_decay_tau = dbl("funnel_decay_tau", 10.0);

        params_.Q_xy = dbl("Q_xy", 60.0);
        params_.Q_theta = dbl("Q_theta", 10.0);
        params_.Q_xy_terminal = dbl("Q_xy_terminal", 60.0);
        params_.Q_theta_terminal = dbl("Q_theta_terminal", 100.0);

        params_.R_v = dbl("R_v", 5.0);
        params_.R_omega = dbl("R_omega", 5.0);
        params_.R_rate_v = dbl("R_rate_v", 2.0);
        params_.R_rate_omega = dbl("R_rate_omega", 5.0);

        params_.seg_advance_t = dbl("seg_advance_t", 0.7);
        params_.d_deadband = dbl("d_deadband", 0.02);
        params_.v_error_gain = dbl("v_error_gain", 3.0);
        params_.v_min_scale = dbl("v_min_scale", 0.2);
        params_.blend_alpha = dbl("blend_alpha", 0.7);
        params_.heading_scale_k = dbl("heading_scale_k", 1.0);
        params_.path_reset_threshold = dbl("path_reset_threshold", 0.5);
        params_.goal_threshold = dbl("goal_threshold", 0.3);
        params_.fallback_decay = dbl("fallback_decay", 0.8);

        declare_parameter("map_frame", std::string("map"));
        frame_id_ = get_parameter("map_frame").as_string();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Publish helpers
    // ─────────────────────────────────────────────────────────────────────────

    void publishTwist(const mpc::Control& u)
    {
        geometry_msgs::msg::Twist m;
        m.linear.x = u.v;
        m.angular.z = u.omega;
        cmd_pub_->publish(m);
    }

    void publishRobotPose(const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::PoseStamped m;
        m.header.stamp = stamp;
        m.header.frame_id = frame_id_;
        m.pose.position.x = x_sim_.x;
        m.pose.position.y = x_sim_.y;
        m.pose.orientation = yawToQuat(x_sim_.theta);
        robot_pose_pub_->publish(m);
    }

    void publishFullPath(const rclcpp::Time& stamp)
    {
        nav_msgs::msg::Path m;
        m.header.stamp = stamp;
        m.header.frame_id = frame_id_;
        for (const auto& pt : path_.pts)
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = m.header;
            ps.pose.position.x = pt.pos.x();
            ps.pose.position.y = pt.pos.y();
            ps.pose.orientation.w = 1.0;
            m.poses.push_back(ps);
        }
        path_pub_->publish(m);
    }

    nav_msgs::msg::Path toNavPath(
        const std::vector<mpc::State>& states,
        const rclcpp::Time& stamp) const
    {
        nav_msgs::msg::Path m;
        m.header.stamp = stamp;
        m.header.frame_id = frame_id_;
        m.poses.reserve(states.size());
        for (const auto& s : states)
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = m.header;
            ps.pose.position.x = s.x;
            ps.pose.position.y = s.y;
            ps.pose.orientation = yawToQuat(s.theta);
            m.poses.push_back(ps);
        }
        return m;
    }

    void publishAllDebug(const mpc::DebugInfo& dbg, const rclcpp::Time& stamp)
    {
        if (!dbg.ref_traj.empty())
        {
            ref_pub_->publish(toNavPath(dbg.ref_traj, stamp));
        }
        if (!dbg.pred_traj.empty())
        {
            pred_pub_->publish(toNavPath(dbg.pred_traj, stamp));
        }

        auto pubF64 = [](auto& pub, double val)
        {
            std_msgs::msg::Float64 m;
            m.data = val;
            pub->publish(m);
        };
        pubF64(cte_pub_, dbg.cte_raw);
        pubF64(d_hard_pub_, dbg.d_hard_eff);
        pubF64(v_scale_pub_, dbg.v_scale);
        pubF64(q_theta_pub_, dbg.Q_theta_eff);

        {
            std_msgs::msg::Bool m;
            m.data = dbg.solver_ok;
            solver_ok_pub_->publish(m);
        }
        {
            std_msgs::msg::Bool m;
            m.data = dbg.near_goal;
            near_goal_pub_->publish(m);
        }

        {
            geometry_msgs::msg::PointStamped m;
            m.header.stamp = stamp;
            m.header.frame_id = frame_id_;
            m.point.x = dbg.proj_pt.x();
            m.point.y = dbg.proj_pt.y();
            proj_pub_->publish(m);
        }

        if (!dbg.v_profile.empty())
        {
            std_msgs::msg::Float64MultiArray m;
            m.data.assign(dbg.v_profile.begin(), dbg.v_profile.end());
            v_profile_pub_->publish(m);
        }

        publishCorridorMarkers(dbg, stamp);
        publishNormalMarkers(dbg, stamp);
        publishDiagnostics(dbg, stamp);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Corridor marker array
    //   id 0  LINE_STRIP  left boundary   (orange)
    //   id 1  LINE_STRIP  right boundary  (orange)
    //   id 2  LINE_STRIP  centerline      (green)
    //   id 3  LINE_LIST   rungs           (semi-transparent orange)
    //   id 4  SPHERE_LIST projection pts  (green dots)
    // ─────────────────────────────────────────────────────────────────────────

    void publishCorridorMarkers(
        const mpc::DebugInfo& dbg,
        const rclcpp::Time& stamp)
    {
        const int n = static_cast<int>(dbg.proj_pts.size());
        if (n == 0)
        {
            return;
        }

        auto mkStrip = [&](int id, float r, float g, float b, float a, float lw)
        {
            visualization_msgs::msg::Marker m;
            m.header.stamp = stamp;
            m.header.frame_id = frame_id_;
            m.ns = "corridor";
            m.id = id;
            m.type = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = lw;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;
            m.color.a = a;
            m.pose.orientation.w = 1.0;
            return m;
        };

        auto left_m = mkStrip(0, 1.0f, 0.55f, 0.0f, 1.0f, 0.012f);
        auto right_m = mkStrip(1, 1.0f, 0.55f, 0.0f, 1.0f, 0.012f);
        auto centre_m = mkStrip(2, 0.2f, 0.9f, 0.3f, 0.8f, 0.008f);

        visualization_msgs::msg::Marker rungs;
        rungs.header.stamp = stamp;
        rungs.header.frame_id = frame_id_;
        rungs.ns = "corridor";
        rungs.id = 3;
        rungs.type = visualization_msgs::msg::Marker::LINE_LIST;
        rungs.action = visualization_msgs::msg::Marker::ADD;
        rungs.scale.x = 0.005f;
        rungs.color.r = 1.0f;
        rungs.color.g = 0.55f;
        rungs.color.b = 0.0f;
        rungs.color.a = 0.35f;
        rungs.pose.orientation.w = 1.0;

        visualization_msgs::msg::Marker spheres;
        spheres.header.stamp = stamp;
        spheres.header.frame_id = frame_id_;
        spheres.ns = "corridor";
        spheres.id = 4;
        spheres.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        spheres.action = visualization_msgs::msg::Marker::ADD;
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.025;
        spheres.color.r = 0.2f;
        spheres.color.g = 0.9f;
        spheres.color.b = 0.3f;
        spheres.color.a = 1.0f;
        spheres.pose.orientation.w = 1.0;

        auto pt2d = [](double x, double y)
        {
            geometry_msgs::msg::Point p;
            p.x = x;
            p.y = y;
            p.z = 0.0;
            return p;
        };

        for (int k = 0; k < n; ++k)
        {
            const Eigen::Vector2d& pp = dbg.proj_pts[k];
            const Eigen::Vector2d& nn = dbg.seg_normals[k];
            const double h = dbg.d_hard_eff;

            left_m.points.push_back(
                pt2d(pp.x() + h * nn.x(), pp.y() + h * nn.y()));
            right_m.points.push_back(
                pt2d(pp.x() - h * nn.x(), pp.y() - h * nn.y()));
            centre_m.points.push_back(pt2d(pp.x(), pp.y()));
            rungs.points.push_back(
                pt2d(pp.x() + h * nn.x(), pp.y() + h * nn.y()));
            rungs.points.push_back(
                pt2d(pp.x() - h * nn.x(), pp.y() - h * nn.y()));
            spheres.points.push_back(pt2d(pp.x(), pp.y()));
        }

        visualization_msgs::msg::MarkerArray ma;
        ma.markers = {left_m, right_m, centre_m, rungs, spheres};
        corridor_pub_->publish(ma);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Segment-normal arrows (every other step to reduce clutter)
    // ─────────────────────────────────────────────────────────────────────────

    void publishNormalMarkers(
        const mpc::DebugInfo& dbg,
        const rclcpp::Time& stamp)
    {
        visualization_msgs::msg::MarkerArray ma;

        // Wipe stale arrows from a previous, longer horizon
        visualization_msgs::msg::Marker del;
        del.header.stamp = stamp;
        del.header.frame_id = frame_id_;
        del.ns = "seg_normals";
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        ma.markers.push_back(del);

        const int n = static_cast<int>(dbg.proj_pts.size());
        for (int k = 0; k < n; k += 2)
        {
            const Eigen::Vector2d& pp = dbg.proj_pts[k];
            const Eigen::Vector2d& nn = dbg.seg_normals[k];

            visualization_msgs::msg::Marker arrow;
            arrow.header.stamp = stamp;
            arrow.header.frame_id = frame_id_;
            arrow.ns = "seg_normals";
            arrow.id = k;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.scale.x = 0.006;
            arrow.scale.y = 0.012;
            arrow.scale.z = 0.012;
            arrow.color.b = 1.0f;
            arrow.color.a = 0.75f;
            arrow.pose.orientation.w = 1.0;

            geometry_msgs::msg::Point tail, tip;
            tail.x = pp.x();
            tail.y = pp.y();
            tail.z = 0.0;
            tip.x = pp.x() + 0.04 * nn.x();
            tip.y = pp.y() + 0.04 * nn.y();
            tip.z = 0.0;
            arrow.points = {tail, tip};
            ma.markers.push_back(arrow);
        }

        normals_pub_->publish(ma);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Diagnostics bundle  (all scalars in one message for the Foxglove table)
    // ─────────────────────────────────────────────────────────────────────────

    void publishDiagnostics(
        const mpc::DebugInfo& dbg,
        const rclcpp::Time& /*stamp*/)
    {
        diagnostic_msgs::msg::DiagnosticStatus ds;
        ds.name = "mpc_controller";
        ds.hardware_id = get_name();
        ds.level = dbg.solver_ok
                       ? diagnostic_msgs::msg::DiagnosticStatus::OK
                       : diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        ds.message = dbg.solver_ok ? "Solver OK" : "Solver FAILED";

        auto kv = [&](const std::string& key, double val)
        {
            diagnostic_msgs::msg::KeyValue pair;
            pair.key = key;
            pair.value = std::to_string(val);
            ds.values.push_back(pair);
        };

        kv("sim_time", sim_time_);
        kv("step", static_cast<double>(step_count_));
        kv("paused", paused_ ? 1.0 : 0.0);
        kv("scenario", static_cast<double>(scenario_));
        kv("robot.x", x_sim_.x);
        kv("robot.y", x_sim_.y);
        kv("robot.theta", x_sim_.theta);
        kv("cte_raw", dbg.cte_raw);
        kv("d_hard_eff", dbg.d_hard_eff);
        kv("v_scale", dbg.v_scale);
        kv("Q_theta_eff", dbg.Q_theta_eff);
        kv("solver_ok", dbg.solver_ok ? 1.0 : 0.0);
        kv("near_goal", dbg.near_goal ? 1.0 : 0.0);
        if (!dbg.v_profile.empty())
        {
            kv("v_profile[0]", dbg.v_profile.front());
            kv("v_profile[N]", dbg.v_profile.back());
        }

        diag_pub_->publish(ds);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Members
    // ─────────────────────────────────────────────────────────────────────────

    // MPC
    mpc::MPCParams params_;
    std::unique_ptr<mpc::MPCController> controller_;

    // Simulation state
    mpc::Path path_;
    mpc::State x_sim_{0.0, 0.0, 0.0};
    double sim_time_ = 0.0;
    int step_count_ = 0;
    int scenario_ = 0;
    bool paused_ = false;
    std::string frame_id_{"map"};

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // Subscriptions
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;

    // Services
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pause_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr restart_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ctrl_reset_srv_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ref_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cte_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr d_hard_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr v_scale_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr q_theta_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr solver_ok_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr near_goal_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr proj_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
        v_profile_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
        robot_pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
        corridor_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
        normals_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr
        diag_pub_;
};

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPCNode>());
    rclcpp::shutdown();
    return 0;
}
