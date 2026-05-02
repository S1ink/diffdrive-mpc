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
#include <cstddef>

namespace mpc
{

// ── Debug / visualisation snapshot ───────────────────────────────────────────

struct DebugInfo
{
    // ── Trajectories ──────────────────────────────────────────────────

    /// Predicted state trajectory in global frame (x,y,θ), converted from
    /// the Frenet solution.  Empty if the solver failed.
    std::vector<State> pred_traj;

    /// Path centreline at the predicted s_k positions.  Used to visualise
    /// the reference path in Foxglove as a nav_msgs/Path.
    std::vector<State> ref_traj;

    // ── Corridor geometry ─────────────────────────────────────────────

    /// Unit path normals n(s_k) at each horizon step.  Length N+1.
    std::vector<Eigen::Vector2d> seg_normals;

    /// Path positions p(s_k) at each horizon step (corridor centre).
    /// Length N+1.
    std::vector<Eigen::Vector2d> proj_pts;

    /// Closest point on path to the (latency-compensated) robot position.
    Eigen::Vector2d proj_pt = Eigen::Vector2d::Zero();

    // ── Velocity profile ──────────────────────────────────────────────

    /// Reference speed at each horizon step [m/s].
    std::vector<double> v_profile;

    // ── Scalars ───────────────────────────────────────────────────────

    double d_hard_eff = 0.0;  // effective corridor width at k=0 [m]
    double cte_raw = 0.0;     // = e_y_0, initial cross-track error [m]
    double v_scale = 1.0;     // legacy field kept for node compatibility
    double Q_theta_eff = 0.0; // legacy field kept for node compatibility

    // ── Status ────────────────────────────────────────────────────────

    bool solver_ok = false;
    bool near_goal = false;
    double solve_ms = 0.0;
    size_t proj_segment_index = 0;
};

// ── Controller ────────────────────────────────────────────────────────────────

class MPCController
{
public:
    explicit MPCController(const MPCParams& p);

    Control update(const State& x_measured, const Path& path);

    void reset();

    size_t pruneTraversedSegments(Path& path);

    const DebugInfo& debugInfo() const { return debug_info_; }

private:
    MPCParams params_;
    Projector projector_;
    ReferenceGenerator ref_gen_;
    Linearizer linearizer_;
    QPBuilder qp_builder_;
    Solver solver_;

    Control u_prev_{0.0, 0.0};
    size_t path_hash_{0};
    bool initialized_{false};

    // Warm-start Frenet trajectory from the most recent successful solve.
    // Length N+1.  Shifted by one step each cycle like the OSQP primal.
    std::vector<FrenetState> frenet_warm_start_;

    DebugInfo debug_info_;

    State latencyCompensate(const State& x) const;
    bool nearGoal(double s0, double total_arc, double e_y_0) const;
    double distToEnd(const Path& path, const ProjectionResult& proj) const;
    static size_t hashPath(const Path& path);

    /// Convert a Frenet prediction back to global (x,y,θ) for visualisation.
    static State frenetToGlobal(
        const FrenetState& fs,
        const Path& path,
        const std::vector<double>& cum);
};

}  // namespace mpc
