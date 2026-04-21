#pragma once

#include "params.hpp"
#include "types.hpp"
#include "path.hpp"
#include "projection.hpp"
#include "reference.hpp"
#include "linearization.hpp"
#include "qp_builder.hpp"
#include "solver.hpp"

#include <Eigen/Dense>
#include <vector>

namespace mpc
{

// ── Debug / visualisation snapshot ───────────────────────────────────────────

/// Everything a visualizer / ROS publisher needs from one MPC cycle.
/// Populated unconditionally by MPCController::update(); read via debugInfo().
struct DebugInfo
{
    // ── Trajectories ──────────────────────────────────────────────────

    /// MPC-predicted state trajectory x_0 … x_N (from OSQP solution).
    /// Empty if the solver failed.
    std::vector<State> pred_traj;

    /// Blended reference trajectory x_0 … x_N used to build the QP.
    std::vector<State> ref_traj;

    // ── Corridor geometry ─────────────────────────────────────────────

    /// Unit normals of the assigned corridor segment for each horizon step.
    /// Length N+1.  Used to draw corridor walls in Foxglove.
    std::vector<Eigen::Vector2d> seg_normals;

    /// Closest point on the path for each horizon step (corridor centre).
    /// Length N+1.
    std::vector<Eigen::Vector2d> proj_pts;

    /// Closest point on path to the (latency-compensated) robot position.
    Eigen::Vector2d proj_pt = Eigen::Vector2d::Zero();

    // ── Velocity profile ──────────────────────────────────────────────

    /// Reference speed at each horizon step [m/s].  Length N+1.
    /// Incorporates curvature limit, braking limit, and v_scale.
    std::vector<double> v_profile;

    // ── Adaptive scalars ──────────────────────────────────────────────

    /// Effective corridor half-width [m] used this cycle.
    /// May be widened relative to params.d_hard during recovery.
    double d_hard_eff = 0.0;

    /// Raw (un-deadbanded) signed cross-track error [m].
    /// Positive = robot is to the left of the path.
    double cte_raw = 0.0;

    /// Velocity reduction scale factor this cycle.
    /// v_scale = clamp(1 − v_error_gain·|cte|,  v_min_scale, 1).
    double v_scale = 1.0;

    /// Effective heading weight Q_theta * exp(−heading_scale_k * |cte|).
    /// Lower when the robot is far from the path so it converges laterally
    /// before aligning heading.
    double Q_theta_eff = 0.0;

    // ── Status flags ──────────────────────────────────────────────────

    /// true if OSQP reported OSQP_SOLVED this cycle.
    bool solver_ok = false;

    /// true when remaining path length < params.goal_threshold.
    bool near_goal = false;
};

// ── Controller ────────────────────────────────────────────────────────────────

/// MPCController — single entry point for all MPC logic.
///
/// Owns all sub-components and implements the adaptive / robustness
/// behaviours described in the Robustness Upgrade Plan:
///
///   A)  Always trust measured state (latency compensation only).
///   B)  Persistent OSQP + warm start (delegated to Solver).
///   C)  Projection reset on large path changes.
///       Reference blending across path updates.
///       Goal clamping.
///   D)  Adaptive corridor width.
///       Velocity reduction under cross-track error.
///       Cross-track deadband.
///   E)  Terminal velocity constraint (via QPContext).
///   F)  Heading weight scaling when far from path.
///   G)  Solver failure fallback (u_prev * fallback_decay).
///
/// Usage:
///   MPCController ctrl(params);
///   Control u = ctrl.update(x_measured, path);

class MPCController
{
public:
    explicit MPCController(const MPCParams& p);

    /// Run one MPC cycle.
    ///
    /// @param x_measured   Latest odometry / localisation state.
    /// @param path         Current path polyline (may change every cycle).
    /// @return             Control to apply immediately.
    Control update(const State& x_measured, const Path& path);

    /// Hard reset: clears all internal state (warm start, blended reference,
    /// previous control).  Call after large teleportations or emergency stops.
    void reset();

    /// Read-only access to the debug snapshot from the most recent update().
    const DebugInfo& debugInfo() const { return debug_info_; }

private:
    // ── Sub-components ────────────────────────────────────────────────
    MPCParams params_;
    Projector projector_;
    ReferenceGenerator ref_gen_;
    Linearizer linearizer_;
    QPBuilder qp_builder_;
    Solver solver_;

    // ── Cross-iteration state ─────────────────────────────────────────
    Control u_prev_{0.0, 0.0};
    Reference ref_prev_;
    bool has_prev_ref_{false};

    // ── Debug snapshot (updated every cycle) ──────────────────────────
    DebugInfo debug_info_;

    // ── Helpers ───────────────────────────────────────────────────────

    /// Predict robot state one timestep ahead to compensate command latency.
    State latencyCompensate(const State& x) const;

    /// Signed cross-track distance from x to the nearest projected point.
    double crossTrackError(
        const State& x,
        const ProjectionResult& proj,
        const Path& path) const;

    /// True when the remaining path length is below params_.goal_threshold.
    bool nearGoal(const Path& path, const ProjectionResult& proj) const;

    /// Remaining arc length from (proj.segment_index, proj.t) to path end.
    double distToEnd(const Path& path, const ProjectionResult& proj) const;

    /// Blend two references:  α·r_new + (1−α)·r_old.
    Reference blend(
        const Reference& r_new,
        const Reference& r_old,
        double alpha) const;
};

}  // namespace mpc