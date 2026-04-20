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
}

// ── Main control cycle ────────────────────────────────────────────────────────

Control MPCController::update(const State& x_measured, const Path& path)
{
    if (!path.valid())
    {
        return {0.0, 0.0};
    }

    // ── A. Trust measured state; predict forward by one dt ────────────
    const State x_pred = latencyCompensate(x_measured);

    // ── C1. Projection — forward-only with hysteresis ─────────────────
    // On the first call, or when the path has changed and the robot's
    // projected position is far from the nearest point on the new path,
    // reset the projector so the segment search restarts from the beginning.
    if (!has_prev_ref_)
    {
        projector_.reset();
    }
    else
    {
        // Quick path-change check: project onto new path without hysteresis
        // to see how far we'd jump.  A cheap single-pass is fine here.
        Projector tmp_proj;
        tmp_proj.look_ahead = path.size();  // unrestricted scan
        const ProjectionResult new_p = tmp_proj.project(x_pred, path);
        const double jump = (new_p.proj - ref_prev_.proj_pts[0]).norm();
        if (jump > params_.path_reset_threshold)
        {
            projector_.reset();
        }
    }

    const ProjectionResult proj = projector_.project(x_pred, path);

    // ── D1+D3. Cross-track error with deadband ─────────────────────────
    double cte = crossTrackError(x_pred, proj, path);
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

    ref_prev_ = new_ref;  // store un-blended as base for next iteration
    has_prev_ref_ = true;

    // ── D1. Adaptive corridor width ────────────────────────────────────
    double d_hard_eff = params_.d_hard;
    if (std::abs(cte) > params_.d_hard)
    {
        d_hard_eff = params_.d_hard * params_.adaptive_corridor_scale;
    }

    // ── F1. Adaptive heading weight ────────────────────────────────────
    // Reduce heading penalty when far off path to allow the robot to
    // prioritise lateral convergence over heading alignment.
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
    ctx.Q_theta_eff = Q_theta_eff;
    ctx.Q_theta_terminal_eff = Q_theta_terminal_eff;
    ctx.near_goal = near;

    // ── H. Linearise around reference trajectory ──────────────────────
    // Linearizer::linearize(traj, u) expects:
    //   traj.size() == N   (operating points x_0 … x_{N-1})
    //   u.size()    == N   (operating controls  u_0 … u_{N-1})
    // We use the blended reference states and the v_profile as the
    // nominal forward speed (ω = 0 as nominal since path is smooth).
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
    // Left normal of travel direction
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
        // Position and heading
        out.x_ref[k].x = alpha * r_new.x_ref[k].x + beta * r_old.x_ref[k].x;
        out.x_ref[k].y = alpha * r_new.x_ref[k].y + beta * r_old.x_ref[k].y;

        // Heading: blend via angle averaging (handles wrap-around)
        const double th_new = r_new.x_ref[k].theta;
        const double th_old = r_old.x_ref[k].theta;
        double d_th = th_new - th_old;
        // Wrap d_th to (−π, π]
        while (d_th > M_PI)
        {
            d_th -= 2.0 * M_PI;
        }
        while (d_th < -M_PI)
        {
            d_th += 2.0 * M_PI;
        }
        out.x_ref[k].theta = th_old + alpha * d_th;

        // Corridor geometry (don't blend normals — use the new path's geometry
        // for the constraint, only the reference point is blended)
        out.seg_normals[k] = r_new.seg_normals[k];
        out.proj_pts[k] = alpha * r_new.proj_pts[k] + beta * r_old.proj_pts[k];

        // Velocity profile
        out.v_profile[k] =
            alpha * r_new.v_profile[k] + beta * r_old.v_profile[k];
    }

    return out;
}

}  // namespace mpc
