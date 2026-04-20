#pragma once

#include "params.hpp"
#include "types.hpp"
#include "path.hpp"
#include "projection.hpp"
#include "reference.hpp"
#include "linearization.hpp"
#include "qp_builder.hpp"
#include "solver.hpp"

namespace mpc
{

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
