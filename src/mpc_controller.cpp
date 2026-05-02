#include "mpc/mpc_controller.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <functional>

namespace mpc
{

// ── Construction / reset ──────────────────────────────────────────────────────

MPCController::MPCController(const MPCParams& p) :
    params_(p),
    projector_(),
    ref_gen_(p),
    linearizer_(p.dt),
    qp_builder_(p)
{
    const int n_brake = params_.minBrakingSteps();
    if (params_.N < n_brake)
    {
        std::cerr << "[MPCController] N=" << params_.N
                  << " < minBrakingSteps()=" << n_brake
                  << "; raising to " << n_brake << ".\n";
        params_.N = n_brake;
        ref_gen_ = ReferenceGenerator(params_);
        qp_builder_ = QPBuilder(params_);
    }
}

void MPCController::reset()
{
    projector_.reset();
    u_prev_ = {0.0, 0.0};
    initialized_ = false;
    path_hash_ = 0;
    frenet_warm_start_.clear();
    debug_info_ = DebugInfo{};
}

// ── pruneTraversedSegments ────────────────────────────────────────────────────

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
    return n_remove;
}

// ── hashPath ──────────────────────────────────────────────────────────────────

size_t MPCController::hashPath(const Path& path)
{
    size_t h = std::hash<size_t>{}(path.size());
    for (const auto& pt : path.pts)
    {
        h ^= std::hash<double>{}(pt.pos.x()) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(pt.pos.y()) + 0x9e3779b9u + (h << 6) + (h >> 2);
    }
    return h;
}

// ── frenetToGlobal ────────────────────────────────────────────────────────────

State MPCController::frenetToGlobal(
    const FrenetState& fs,
    const Path& path,
    const std::vector<double>& cum)
{
    const int seg = segmentAtArc(cum, fs.s);
    const double seg_len = cum[seg + 1] - cum[seg];
    const double t =
        (seg_len > 1e-12)
            ? std::clamp((fs.s - cum[seg]) / seg_len, 0.0, 1.0)
            : 0.0;

    const Eigen::Vector2d path_pos =
        path.pts[seg].pos + t * (path.pts[seg + 1].pos - path.pts[seg].pos);
    const Eigen::Vector2d dir = path.segmentDir(seg);
    const Eigen::Vector2d n(-dir.y(), dir.x());
    const double theta_path = std::atan2(dir.y(), dir.x());

    return {
        path_pos.x() + fs.e_y * n.x(),
        path_pos.y() + fs.e_y * n.y(),
        theta_path + fs.e_theta};
}

// ── Helpers ───────────────────────────────────────────────────────────────────

State MPCController::latencyCompensate(const State& x) const
{
    return {
        x.x + u_prev_.v * std::cos(x.theta) * params_.dt,
        x.y + u_prev_.v * std::sin(x.theta) * params_.dt,
        x.theta + u_prev_.omega * params_.dt};
}

double MPCController::distToEnd(
    const Path& path,
    const ProjectionResult& proj) const
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

bool MPCController::nearGoal(
    double /*s0*/,
    double /*total_arc*/,
    double e_y_0) const
{
    return debug_info_.near_goal &&
           (std::abs(e_y_0) < params_.d_hard * params_.goal_cte_scale);
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

    const int N = params_.N;
    const double dt = params_.dt;

    // ── A. Latency compensation ───────────────────────────────────────
    const State x_pred = latencyCompensate(x_measured);

    // ── B. Path change detection ──────────────────────────────────────
    const size_t new_hash = hashPath(path);
    const bool path_changed = !initialized_ || (new_hash != path_hash_);
    if (path_changed)
    {
        projector_.reset();
        frenet_warm_start_.clear();
    }
    path_hash_ = new_hash;
    initialized_ = true;

    // ── C. Project onto path ──────────────────────────────────────────
    const ProjectionResult proj = projector_.project(x_pred, path);

    // ── D. Arc-length data ────────────────────────────────────────────
    const std::vector<double> cum = cumulativeArcs(path);
    const double total_arc = cum.back();
    const int n_pts = (int)path.size();

    // Segment lengths
    std::vector<double> segl(n_pts - 1);
    for (int i = 0; i < n_pts - 1; ++i)
    {
        segl[i] = cum[i + 1] - cum[i];
    }

    // Arc-length of current projection
    const size_t seg0 = proj.segment_index;
    const double s0 = cum[seg0] + proj.t * segl[seg0];

    // ── E. Compute initial Frenet state ───────────────────────────────
    const Eigen::Vector2d dir0 = path.segmentDir(seg0);
    const double theta_path0 = std::atan2(dir0.y(), dir0.x());
    const Eigen::Vector2d n0(-dir0.y(), dir0.x());

    const double e_y_0 =
        n0.dot(Eigen::Vector2d(x_pred.x, x_pred.y) - proj.proj);

    double e_theta_0 = x_pred.theta - theta_path0;
    while (e_theta_0 > M_PI)  { e_theta_0 -= 2.0 * M_PI; }
    while (e_theta_0 < -M_PI) { e_theta_0 += 2.0 * M_PI; }

    const FrenetState x0_frenet = {s0, e_y_0, e_theta_0};

    // ── F. Velocity profile ───────────────────────────────────────────
    const Reference ref = ref_gen_.generate(path, proj, u_prev_.v);

    // ── G. Near-goal detection ────────────────────────────────────────
    const double remaining = distToEnd(path, proj);
    const bool near =
        (remaining < params_.goal_threshold) &&
        (std::abs(e_y_0) < params_.d_hard * params_.goal_cte_scale);

    // ── H. Initialise / advance warm-start Frenet trajectory ─────────
    //
    // On path change or first call, seed with a straight-ahead trajectory
    // at the velocity profile speed.  Otherwise use the shifted solution
    // from the previous cycle and clamp s to [s0, total_arc].
    if ((int)frenet_warm_start_.size() != N + 1)
    {
        frenet_warm_start_.resize(N + 1);
        frenet_warm_start_[0] = x0_frenet;
        for (int k = 1; k <= N; ++k)
        {
            const double v_k = ref.v_profile[std::min(k - 1, N - 1)];
            frenet_warm_start_[k] = {
                std::min(s0 + k * v_k * dt, total_arc),
                0.0,
                0.0};
        }
    }
    else
    {
        // Override the initial state with the current measurement.
        frenet_warm_start_[0] = x0_frenet;
        for (int k = 1; k <= N; ++k)
        {
            frenet_warm_start_[k].s =
                std::clamp(frenet_warm_start_[k].s, s0, total_arc);
        }
    }

    // ── I. Compute delta_theta_path for linearisation ─────────────────
    //
    // delta_theta_path[k] = θ_path(s̄_{k+1}) − θ_path(s̄_k).
    // Zero within a segment; equals the signed corner angle at junctions.
    // This encodes path-heading discontinuities exactly in the affine
    // residual d_k without any additional geometric logic in the QP.
    std::vector<double> delta_theta(N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        const double sk = frenet_warm_start_[k].s;
        const double sk1 = frenet_warm_start_[k + 1].s;
        const int seg_k = segmentAtArc(cum, sk);
        const int seg_k1 = segmentAtArc(cum, sk1);
        if (seg_k != seg_k1)
        {
            const Eigen::Vector2d d_k = path.segmentDir(seg_k);
            const Eigen::Vector2d d_k1 = path.segmentDir(seg_k1);
            double dth =
                std::atan2(d_k1.y(), d_k1.x()) -
                std::atan2(d_k.y(), d_k.x());
            while (dth > M_PI)  { dth -= 2.0 * M_PI; }
            while (dth < -M_PI) { dth += 2.0 * M_PI; }
            delta_theta[k] = dth;
        }
    }

    // ── J. Linearise around warm-start ────────────────────────────────
    std::vector<FrenetState> lin_traj(N);
    std::vector<Control> lin_ctrl(N);
    for (int k = 0; k < N; ++k)
    {
        lin_traj[k] = frenet_warm_start_[k];
        lin_ctrl[k] = {ref.v_profile[k], 0.0};
    }
    const LinModel model = linearizer_.linearize(lin_traj, lin_ctrl, delta_theta);

    // ── K. Build QP ───────────────────────────────────────────────────
    QPContext ctx;
    ctx.e_y_0 = e_y_0;
    ctx.near_goal = near;

    const QP qp = qp_builder_.build(x0_frenet, u_prev_, ref, model, ctx);
    const bool ok = solver_.update(qp, N);

    // ── L. Extract solution and update warm start ──────────────────────
    if (ok)
    {
        const std::vector<FrenetState> frenet_pred = solver_.getStatePrediction();
        frenet_warm_start_ = frenet_pred;

        // Convert Frenet prediction to global for visualisation.
        const int np = (int)frenet_pred.size();
        debug_info_.pred_traj.resize(np);
        debug_info_.ref_traj.resize(np);
        debug_info_.seg_normals.resize(np);
        debug_info_.proj_pts.resize(np);

        for (int k = 0; k < np; ++k)
        {
            const FrenetState& fs = frenet_pred[k];
            debug_info_.pred_traj[k] = frenetToGlobal(fs, path, cum);

            // Path centreline and normal at s_k (for corridor visualisation).
            const int seg_k = segmentAtArc(cum, fs.s);
            const double sl = cum[seg_k + 1] - cum[seg_k];
            const double t_k =
                (sl > 1e-12)
                    ? std::clamp((fs.s - cum[seg_k]) / sl, 0.0, 1.0)
                    : 0.0;
            const Eigen::Vector2d pp =
                path.pts[seg_k].pos +
                t_k * (path.pts[seg_k + 1].pos - path.pts[seg_k].pos);
            const Eigen::Vector2d dir_k = path.segmentDir(seg_k);
            const Eigen::Vector2d nk(-dir_k.y(), dir_k.x());
            const double th_k = std::atan2(dir_k.y(), dir_k.x());

            debug_info_.proj_pts[k] = pp;
            debug_info_.seg_normals[k] = nk;
            debug_info_.ref_traj[k] = {pp.x(), pp.y(), th_k};
        }
    }
    else
    {
        debug_info_.pred_traj.clear();
    }

    // ── M. Populate debug snapshot ─────────────────────────────────────
    const double d0 =
        params_.d_hard +
        std::max(0.0, std::abs(e_y_0) - params_.d_hard);

    debug_info_.solver_ok = ok;
    debug_info_.cte_raw = e_y_0;
    debug_info_.d_hard_eff = d0;
    debug_info_.v_scale = 1.0;
    debug_info_.Q_theta_eff = params_.Q_eth;
    debug_info_.near_goal = near;
    debug_info_.proj_pt = proj.proj;
    debug_info_.proj_segment_index = proj.segment_index;
    debug_info_.v_profile = ref.v_profile;

    debug_info_.solve_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0)
            .count();

    // ── N. Failure fallback ────────────────────────────────────────────
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

}  // namespace mpc
