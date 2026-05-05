#include "mpc/mpc_controller.hpp"

#include <cmath>
#include <chrono>
#include <cassert>
#include <iostream>
#include <algorithm>


namespace mpc
{

// ── Construction / reset ─────────────────────────────────────────────────────

MPCController::MPCController(const MPCParams& p) :
    params_(p),
    projector_(),
    ref_gen_(p),
    linearizer_(p.dt),
    qp_builder_(p)
{
    // Ensure the horizon covers the complete maximum-braking distance.
    //
    // The reference generator's backward-pass velocity profile guarantees
    // that v_profile[k] already accounts for all upcoming speed limits, but
    // only for events that lie within the horizon.  If N < minBrakingSteps()
    // a corner whose braking would need to begin BEFORE the first horizon
    // step is invisible to the planner: the robot would enter it too fast.
    //
    // We raise N here (never lower it) so that no matter what the caller set
    // in MPCParams, the solver always looks far enough ahead.
    const int n_brake = params_.minBrakingSteps();
    if (params_.N < n_brake)
    {
        std::cerr << "[MPCController] N=" << params_.N
                  << " is smaller than minBrakingSteps()=" << n_brake
                  << "; raising N to " << n_brake << ".\n";
        params_.N = n_brake;
        // Propagate the updated N into the sub-components that hold their own
        // copy of MPCParams.
        ref_gen_ = ReferenceGenerator(params_);
        qp_builder_ = QPBuilder(params_);
    }
}

void MPCController::reset()
{
    projector_.reset();
    u_prev_ = {0.0, 0.0};
    has_prev_ref_ = false;
    path_hash_ = 0;
    debug_info_ = DebugInfo{};
}

// ── pruneTraversedSegments ────────────────────────────────────────────────────────

size_t MPCController::pruneTraversedSegments(Path& path)
{
    const size_t n_remove = debug_info_.proj_segment_index;
    if (n_remove == 0 || path.size() <= n_remove + 1)
    {
        return 0;
    }
    path.pts.erase(
        path.pts.begin(),
        path.pts.begin() + static_cast<std::ptrdiff_t>(n_remove));
    // Path hash changes next cycle → projector_.reset() runs automatically.
    return n_remove;
}

// ── Path hash ─────────────────────────────────────────────────────────────────

size_t MPCController::hashPath(const Path& path)
{
    // FNV-1a-inspired mix over all waypoint coordinates.
    // Fast and collision-resistant enough for detecting path identity changes.
    size_t h = std::hash<size_t>{}(path.size());
    for (const auto& pt : path.pts)
    {
        h ^=
            std::hash<double>{}(pt.pos.x()) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^=
            std::hash<double>{}(pt.pos.y()) + 0x9e3779b9u + (h << 6) + (h >> 2);
    }
    return h;
}

// ── Main control cycle ────────────────────────────────────────────────────────

Control MPCController::update(const State& x_measured, const Path& path)
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    if (!path.valid())
    {
        debug_info_ = DebugInfo{};
        return {0.0, 0.0};
    }

    // ── A. Trust measured state; predict forward by one dt ────────────
    const State x_pred = latencyCompensate(x_measured);

    // ── C1. Path change detection (hash-based) ────────────────────────
    //
    // Compare the full path geometry each cycle.  On any change we reset
    // the projector so segment tracking restarts from 0 via bisector
    // traversal.  This is more reliable than the previous jump-distance
    // heuristic, which could miss in-place path modifications and trigger
    // spuriously on normal path progress.
    //
    // v_cur (= u_prev_.v) is robot state and is intentionally preserved
    // across path changes so the velocity profile seeds from a realistic
    // initial speed.
    const size_t new_hash = hashPath(path);
    const bool path_changed = !has_prev_ref_ || (new_hash != path_hash_);
    if (path_changed)
    {
        projector_.reset();
    }
    path_hash_ = new_hash;

    const ProjectionResult proj = projector_.project(x_pred, path);

    // ── C2. Generate reference, seeded with the robot's current speed ──
    //
    // Passing u_prev_.v as v_cur seeds the look-ahead braking integrator
    // at the actual robot speed, giving a kinematically-continuous profile.
    // new_ref.cte is measured against the smooth arc rather than the raw
    // polyline, so corners don't produce spuriously large CTE values.
    Reference new_ref = ref_gen_.generate(
        path,
        new_hash,
        Eigen::Vector2d(x_pred.x, x_pred.y),
        u_prev_.v);

    // ── D1+D3. Cross-track error with deadband ─────────────────────────
    const double cte_raw = new_ref.cte;

    // ── Fix 1: Stanley heading correction ────────────────────────────
    //
    // The reference heading from the generator always points along the
    // path tangent.  When the robot is off-path this is wrong: the
    // optimizer needs a target that first rotates the robot TOWARD the
    // path, not one that tells it to drive parallel to a path it cannot
    // yet reach.
    //
    // Stanley steering formula:
    //   θ_ref_k += atan2(−stanley_k · cte_raw,  max(v_k, v_min))
    //
    // Properties:
    //   cte = 0      → correction = 0  (pure path-tangent, no change)
    //   cte > 0      → negative correction (rotate right, toward path)
    //   v large      → small correction (gentle at speed, no over-steer)
    //   v → 0        → bounded at ≈ ±atan(stanley_k·cte/v_min) ≤ 90°
    //
    // Applied to new_ref before blending so the blend also interpolates
    // the corrected heading, not just the geometric one.
    {
        const double k = params_.stanley_k;
        const double v_min_st = params_.stanley_v_min;
        const double decay = params_.stanley_decay;
        for (int kk = 0; kk <= params_.N; ++kk)
        {
            const double v_k = std::max(new_ref.v_profile[kk], v_min_st);
            const double correction =
                std::atan2(-k * cte_raw, v_k) * std::exp(-decay * kk);
            double& th = new_ref.x_ref[kk].theta;
            th += correction;
            while (th > M_PI)
            {
                th -= 2.0 * M_PI;
            }
            while (th < -M_PI)
            {
                th += 2.0 * M_PI;
            }
        }
    }

    // ── C2. Blend with previous reference to smooth same-path updates ──
    //
    // Blending is suppressed when the path has changed: applying the old
    // reference geometry to new corridor normals produces incorrect QP
    // constraints, so the new reference is used as-is for that cycle.
    const Reference ref = (has_prev_ref_ && !path_changed)
                              ? blend(new_ref, ref_prev_, params_.blend_alpha)
                              : new_ref;

    ref_prev_ = new_ref;
    has_prev_ref_ = true;

    // ── Normalise reference headings (heading-wrap fix) ─────────────────
    // Moved here so the recovery check below can read corrected heading at k=0.
    Reference ref_qp = ref;
    {
        const double anchor = x_pred.theta;
        for (auto& s : ref_qp.x_ref)
        {
            double d = s.theta - anchor;
            while (d > M_PI)
            {
                d -= 2.0 * M_PI;
            }
            while (d < -M_PI)
            {
                d += 2.0 * M_PI;
            }
            s.theta = anchor + d;
        }
    }

    // ── C3. Near-goal detection ────────────────────────────────────────
    // Terminal-v=0 only when BOTH near the path end AND laterally close.
    const bool near =
        (new_ref.remaining_arc < params_.goal_threshold) &&
        (std::abs(cte_raw) < params_.d_hard * params_.goal_cte_scale);

    // ── Build QPContext ─────────────────────────────────────────
    QPContext ctx;
    ctx.d_hard_eff = params_.d_hard;
    ctx.cte_raw = cte_raw;
    ctx.Q_theta_eff = params_.Q_theta;
    ctx.Q_theta_terminal_eff = params_.Q_theta_terminal;
    ctx.near_goal = near;

    // ── H. Linearise around previous predicted trajectory (SQP step) ──
    const int N = params_.N;
    std::vector<State> lin_traj(N);
    std::vector<Control> lin_ctrl(N);
    {
        const std::vector<State>& prev_pred = debug_info_.pred_traj;
        const bool have_prev_pred = ((int)prev_pred.size() >= N);
        for (int k = 0; k < N; ++k)
        {
            lin_traj[k] = have_prev_pred ? prev_pred[k] : ref.x_ref[k];
            lin_ctrl[k] = {ref.v_profile[k], 0.0};
        }
    }
    const LinModel model = linearizer_.linearize(lin_traj, lin_ctrl);

    // ── Build QP and solve with wall-clock timing ────────────────────
    const QP qp = qp_builder_.build(x_pred, u_prev_, ref_qp, model, ctx);
    const bool ok = solver_.update(qp, N);

    // ── Populate debug snapshot ────────────────────────────────────────
    debug_info_.solver_ok = ok;
    debug_info_.cte_raw = cte_raw;
    debug_info_.near_goal = near;
    debug_info_.proj_pt = proj.proj;
    debug_info_.proj_segment_index = proj.segment_index;
    debug_info_.ref_snap = std::move(ref_qp);
    debug_info_.pred_traj =
        ok ? solver_.getStatePrediction() : std::vector<State>{};

    debug_info_.solve_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - t0)
                               .count();

    // ── G. Failure fallback ────────────────────────────────────────────
    if (!ok)
    {
        u_prev_.v *= params_.fallback_decay;
        u_prev_.omega *= params_.fallback_decay;
        return u_prev_;
    }

    u_prev_ = solver_.getControl();
    return u_prev_;
}

// ── Private helpers ───────────────────────────────────────────────────────────

State MPCController::latencyCompensate(const State& x) const
{
    if (iszero(params_.feedback_delay_s))
    {
        return x;
    }
    return {
        x.x + u_prev_.v * std::cos(x.theta) * params_.feedback_delay_s,
        x.y + u_prev_.v * std::sin(x.theta) * params_.feedback_delay_s,
        x.theta + u_prev_.omega * params_.feedback_delay_s};
}

Reference MPCController::blend(
    const Reference& r_new,
    const Reference& r_old,
    double alpha) const
{
    assert(r_new.x_ref.size() == r_old.x_ref.size());
    const int n = (int)r_new.x_ref.size();

    Reference out;
    out.x_ref.resize(n);
    out.seg_normals.resize(n);
    out.proj_pts.resize(n);
    out.v_profile.resize(n);

    const double beta = 1.0 - alpha;

    for (int k = 0; k < n; ++k)
    {
        out.x_ref[k].x = alpha * r_new.x_ref[k].x + beta * r_old.x_ref[k].x;
        out.x_ref[k].y = alpha * r_new.x_ref[k].y + beta * r_old.x_ref[k].y;

        const double th_new = r_new.x_ref[k].theta;
        const double th_old = r_old.x_ref[k].theta;
        double d_th = th_new - th_old;
        while (d_th > M_PI)
        {
            d_th -= 2.0 * M_PI;
        }
        while (d_th < -M_PI)
        {
            d_th += 2.0 * M_PI;
        }
        // Normalise interpolated heading back to [-pi, pi].
        // Without this, th_old + alpha*d_th can slip outside that range
        // and silently corrupt subsequent cycles.
        {
            double th = th_old + alpha * d_th;
            while (th > M_PI)
            {
                th -= 2.0 * M_PI;
            }
            while (th < -M_PI)
            {
                th += 2.0 * M_PI;
            }
            out.x_ref[k].theta = th;
        }

        // Corridor geometry always from the new path (never blend normals:
        // mixing normals from two different path geometries would corrupt the
        // QP half-space constraints).
        out.seg_normals[k] = r_new.seg_normals[k];
        out.proj_pts[k] = r_new.proj_pts[k];

        out.v_profile[k] =
            alpha * r_new.v_profile[k] + beta * r_old.v_profile[k];
    }

    return out;
}

}  // namespace mpc
