#pragma once

#include "path.hpp"
#include "types.hpp"
#include "params.hpp"
#include "solver.hpp"
#include "reference.hpp"
#include "projection.hpp"
#include "qp_builder.hpp"
#include "linearization.hpp"

#include <Eigen/Dense>

#include <vector>
#include <cstddef>


namespace mpc
{

// Debug snapshot populated each update(). Used for visualization/logging.
struct DebugInfo
{
    std::vector<State> pred_traj;  // predicted state trajectory x_0 ... x_N
    Reference ref_snap;            // reference snapshot passed to QP
    Eigen::Vector2d proj_pt = Eigen::Vector2d::Zero();
    double cte_raw = 0.0;  // signed cross-track error [m], + left
    bool solver_ok = false;
    bool near_goal = false;
    double solve_ms = 0.0;          // OSQP wall-clock time [ms]
    size_t proj_segment_index = 0;  // index of closest path segment
};

// MPCController: main MPC entry point. Owns subcomponents and runs the
// control cycle: latency compensation, reference generation/blending,
// linearization, QP assembly/solve, and fallback on solver failure.
class MPCController
{
public:
    explicit MPCController(const MPCParams& p);

    // Run one MPC cycle.
    Control update(const State& x_measured, const Path& path);

    // Hard reset of internal state.
    void reset();

    // Remove fully-traversed leading segments from path.
    size_t pruneTraversedSegments(Path& path);

    // Read-only access to debug snapshot from last update().
    const DebugInfo& debugInfo() const { return debug_info_; }

private:
    // Sub-components
    MPCParams params_;
    Projector projector_;
    ReferenceGenerator ref_gen_;
    Linearizer linearizer_;
    QPBuilder qp_builder_;
    Solver solver_;

    // Cross-iteration state
    Control u_prev_{0.0, 0.0};
    Reference ref_prev_;
    bool has_prev_ref_{false};
    size_t path_hash_{0};

    // Debug snapshot
    DebugInfo debug_info_;

    // Helpers
    State latencyCompensate(const State& x) const;
    Reference blend(
        const Reference& r_new,
        const Reference& r_old,
        double alpha) const;
    static size_t hashPath(const Path& path);
};

}  // namespace mpc
