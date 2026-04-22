#include "mpc/mpc_controller.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>

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
    projector_.seg_advance_t = p.seg_advance_t;
}

void MPCController::reset()
{
    projector_.reset();
    u_prev_ = {0.0, 0.0};
    has_prev_ref_ = false;
    path_hash_ = 0;
    debug_info_ = DebugInfo{};
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

    // ── D1+D3. Cross-track error with deadband ─────────────────────────
    const double cte_raw = crossTrackError(x_pred, proj, path);
    double cte = cte_raw;
    if (std::abs(cte) < params_.d_deadband)
    {
        cte = 0.0;
    }

    // ── D2. Velocity scale under lateral error ─────────────────────────
    const double v_scale = std::clamp(
        1.0 - params_.v_error_gain * std::abs(cte),
        params_.v_min_scale,
        1.0);

    // ── C2. Generate reference, seeded with the robot's current speed ──
    //
    // Passing u_prev_.v as v_cur seeds the look-ahead braking integrator
    // at the actual robot speed, giving a kinematically-continuous profile.
    Reference new_ref = ref_gen_.generate(path, proj, u_prev_.v);
    for (double& v : new_ref.v_profile)
    {
        v *= v_scale;
    }

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
        for (int kk = 0; kk <= params_.N; ++kk)
        {
            const double v_k = std::max(new_ref.v_profile[kk], v_min_st);
            const double correction = std::atan2(-k * cte_raw, v_k);
            double& th = new_ref.x_ref[kk].theta;
            th += correction;
            // Normalise to [−π, π]
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

    // ── D1. Adaptive corridor width ────────────────────────────────────
    double d_hard_eff = params_.d_hard;
    if (std::abs(cte) > params_.d_hard)
    {
        d_hard_eff = params_.d_hard * params_.adaptive_corridor_scale;
    }

    // ── F1. Adaptive heading weight ────────────────────────────────────
    //
    // heading_scale_k defaults to 0.0 (disabled).  See params.hpp for why
    // suppressing heading weight at large CTE is counterproductive.
    const double Q_theta_eff =
        params_.Q_theta * std::exp(-params_.heading_scale_k * std::abs(cte));
    const double Q_theta_terminal_eff =
        params_.Q_theta_terminal *
        std::exp(-params_.heading_scale_k * std::abs(cte));

    // ── C3. Near-goal detection ────────────────────────────────────────
    const bool near = nearGoal(path, proj);

    // ── Build QPContext ────────────────────────────────────────────────
    QPContext ctx;
    ctx.d_hard_eff = d_hard_eff;
    ctx.cte_raw = cte_raw;
    ctx.Q_theta_eff = Q_theta_eff;
    ctx.Q_theta_terminal_eff = Q_theta_terminal_eff;
    ctx.near_goal = near;

    // ── H. Linearise around previous predicted trajectory (SQP step) ──
    //
    // Linearizing around the reference works well when the robot is close
    // to the path.  When it is not — exactly the situation we are trying
    // to recover from — the unicycle Jacobians evaluated at the reference
    // heading are a poor approximation of the actual dynamics, and the
    // resulting QP produces controls that do not achieve the predicted
    // effect in the real plant.
    //
    // One SQP iteration: use the previous cycle's predicted state
    // trajectory as the linearization point.  When the prediction is
    // close to the actual trajectory (warm-start quality is good) this
    // is significantly more accurate at no extra solver cost.
    // Falls back to ref.x_ref if no valid previous prediction exists
    // (first cycle, or after a solver failure).
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

    // ── Build and solve QP ────────────────────────────────────────────
    const QP qp = qp_builder_.build(x_pred, u_prev_, ref, model, ctx);
    const bool ok = solver_.update(qp, N);

    // ── Populate debug snapshot ────────────────────────────────────────
    debug_info_.solver_ok = ok;
    debug_info_.cte_raw = cte_raw;
    debug_info_.d_hard_eff = d_hard_eff;
    debug_info_.v_scale = v_scale;
    debug_info_.Q_theta_eff = Q_theta_eff;
    debug_info_.near_goal = near;
    debug_info_.proj_pt = proj.proj;
    debug_info_.ref_traj = ref.x_ref;
    debug_info_.seg_normals = ref.seg_normals;
    debug_info_.proj_pts = ref.proj_pts;
    debug_info_.v_profile = ref.v_profile;  // post-v_scale, post-blend
    debug_info_.pred_traj =
        ok ? solver_.getStatePrediction() : std::vector<State>{};

    // ── G. Failure fallback ────────────────────────────────────────────
    if (!ok)
    {
        u_prev_.v *= params_.fallback_decay;
        u_prev_.omega *= params_.fallback_decay;
        return u_prev_;
    }

    const Control u = solver_.getControl();
    u_prev_ = u;
    return u;
}

// ── Private helpers ───────────────────────────────────────────────────────────

State MPCController::latencyCompensate(const State& x) const
{
    return {
        x.x + u_prev_.v * std::cos(x.theta) * params_.dt,
        x.y + u_prev_.v * std::sin(x.theta) * params_.dt,
        x.theta + u_prev_.omega * params_.dt};
}

double MPCController::crossTrackError(
    const State& x,
    const ProjectionResult& proj,
    const Path& path) const
{
    const Eigen::Vector2d p(x.x, x.y);
    const Eigen::Vector2d dir = path.segmentDir(proj.segment_index);
    const Eigen::Vector2d n(-dir.y(), dir.x());
    return n.dot(p - proj.proj);
}

double MPCController::distToEnd(const Path& path, const ProjectionResult& proj)
    const
{
    size_t idx = proj.segment_index;
    if (idx >= path.size() - 1)
    {
        return 0.0;
    }

    const double seg_len = (path.pts[idx + 1].pos - path.pts[idx].pos).norm();
    double d = (1.0 - proj.t) * seg_len;

    for (size_t i = idx + 1; i < path.size() - 1; ++i)
    {
        d += (path.pts[i + 1].pos - path.pts[i].pos).norm();
    }

    return d;
}

bool MPCController::nearGoal(const Path& path, const ProjectionResult& proj)
    const
{
    return distToEnd(path, proj) < params_.goal_threshold;
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
        out.x_ref[k].theta = th_old + alpha * d_th;

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
