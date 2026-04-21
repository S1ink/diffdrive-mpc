#include "mpc/mpc_controller.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

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
    debug_info_ = DebugInfo{};
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

    // ── C1. Projection — forward-only with hysteresis ─────────────────
    if (!has_prev_ref_)
    {
        projector_.reset();
    }
    else
    {
        Projector tmp_proj;
        tmp_proj.look_ahead = path.size();
        const ProjectionResult new_p = tmp_proj.project(x_pred, path);
        const double jump = (new_p.proj - ref_prev_.proj_pts[0]).norm();
        if (jump > params_.path_reset_threshold)
        {
            projector_.reset();
        }
    }

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

    // ── C2. Generate reference + apply velocity reduction ─────────────
    Reference new_ref = ref_gen_.generate(path, proj);
    for (double& v : new_ref.v_profile)
    {
        v *= v_scale;
    }

    // ── C2. Blend with previous reference to smooth path updates ──────
    const Reference ref = has_prev_ref_
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

    // ── H. Linearise around reference trajectory ──────────────────────
    const int N = params_.N;
    std::vector<State> lin_traj(N);
    std::vector<Control> lin_ctrl(N);
    for (int k = 0; k < N; ++k)
    {
        lin_traj[k] = ref.x_ref[k];
        lin_ctrl[k] = {ref.v_profile[k], 0.0};
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

        // Corridor geometry always from the new path
        out.seg_normals[k] = r_new.seg_normals[k];
        out.proj_pts[k] = r_new.proj_pts[k];

        out.v_profile[k] =
            alpha * r_new.v_profile[k] + beta * r_old.v_profile[k];
    }

    return out;
}

}  // namespace mpc
