#include "mpc/reference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mpc
{

// ── File-scope helpers ────────────────────────────────────────────────────────

/// Compute cumulative arc lengths along the path.
/// Returns a vector of length path.size() where cum[i] is the arc length from
/// pts[0] to pts[i].
static std::vector<double> cumulativeArcs(const Path& path)
{
    const int n = (int)path.size();
    std::vector<double> cum(n, 0.0);
    for (int i = 1; i < n; ++i)
    {
        cum[i] = cum[i - 1] + (path.pts[i].pos - path.pts[i - 1].pos).norm();
    }
    return cum;
}

/// Build a path-wide velocity profile using backward-pass deceleration
/// smoothing, matching the approach from control.py::_build_velocity_profile().
///
/// Algorithm
/// ---------
///  1. Set v_lim = v_max everywhere.
///  2. Clamp v_lim to 0 at the path end (terminal stop).
///  3. At each interior waypoint junction, apply the yaw-rate-based corner
///     speed limit:  v ≤ omega_max · ds_avg / angle   (derived from ω=v·κ).
///     Junctions with a heading change < ~3° are skipped as negligible.
///  4. Run three backward-pass sweeps to propagate deceleration constraints:
///         v[i] = min(v[i],  sqrt(v[i+1]² + 2·a_max·ds[i]))
///     Three iterations are enough to resolve cascading constraints (e.g. two
///     tight corners in quick succession where the second corner tightens the
///     entry speed required at the first).
///
/// The result guarantees that at every waypoint the stored speed is the
/// maximum at which the robot CAN travel and still brake to every future
/// speed limit in time under constant deceleration a_max.
///
/// @param path       Path polyline.
/// @param cum        Cumulative arc lengths (length path.size()), from
///                   cumulativeArcs().
/// @param v_max      Maximum allowable forward speed [m/s].
/// @param omega_max  Maximum angular speed used for corner limits [rad/s].
/// @param a_max      Maximum deceleration magnitude [m/s²].
/// @param out_v      Output: v_lim at each waypoint (length path.size()).
static void buildVelocityProfile(
    const Path& path,
    const std::vector<double>& cum,
    double v_max,
    double omega_max,
    double a_max,
    std::vector<double>& out_v)
{
    const int n = (int)path.size();
    out_v.assign(n, v_max);

    // ── Step 2: Terminal stop ─────────────────────────────────────────
    out_v.back() = 0.0;

    // ── Step 3: Corner speed limits ───────────────────────────────────
    //
    // At each interior junction i, the robot's angular velocity is
    //   ω ≈ v · Δθ / ds_avg
    // so ω ≤ omega_max  ⟹  v ≤ omega_max · ds_avg / Δθ.
    //
    // The average of the incoming and outgoing segment lengths is used as
    // the arc estimate ds_avg, which gives a more stable estimate than
    // either segment alone near junctions where one segment may be very short.
    for (int i = 1; i < n - 1; ++i)
    {
        const Eigen::Vector2d d_in =
            (path.pts[i].pos - path.pts[i - 1].pos).normalized();
        const Eigen::Vector2d d_out =
            (path.pts[i + 1].pos - path.pts[i].pos).normalized();

        const double cos_a = std::clamp(d_in.dot(d_out), -1.0, 1.0);
        const double angle = std::acos(cos_a);  // [0, π]

        if (angle < 0.05)
        {
            continue;  // < ~3°: negligible curvature, no limit
        }

        const double l_in = (path.pts[i].pos - path.pts[i - 1].pos).norm();
        const double l_out = (path.pts[i + 1].pos - path.pts[i].pos).norm();
        const double ds_avg = 0.5 * (l_in + l_out);

        const double v_corner =
            std::clamp(omega_max * ds_avg / (angle + 1e-9), 0.0, v_max);
        out_v[i] = std::min(out_v[i], v_corner);
    }

    // ── Step 4: Backward-pass deceleration smoothing (3 sweeps) ──────
    //
    // Each sweep propagates the constraint  v[i] ≤ sqrt(v[i+1]² + 2·a·ds)
    // backward through the path.  One sweep is exact for isolated events;
    // three sweeps ensure convergence when multiple tight turns interact
    // (identical to control.py's loop-of-3 backward pass).
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int i = n - 2; i >= 0; --i)
        {
            const double seg_len =
                (path.pts[i + 1].pos - path.pts[i].pos).norm();
            const double v_reach =
                std::sqrt(out_v[i + 1] * out_v[i + 1] + 2.0 * a_max * seg_len);
            if (v_reach < out_v[i])
            {
                out_v[i] = v_reach;
            }
        }
    }
}

/// Interpolate the precomputed velocity profile at arc position s_abs.
///
/// @param cum    Cumulative arc lengths at each waypoint (sorted, ascending).
/// @param v_lim  Speed limit at each waypoint (parallel to cum).
/// @param s_abs  Query arc position from path start.
/// @param v_max  Fallback cap — returned when s_abs is before the first point.
static double interpVelocityProfile(
    const std::vector<double>& cum,
    const std::vector<double>& v_lim,
    double s_abs,
    double v_max)
{
    if (s_abs >= cum.back())
    {
        return 0.0;  // beyond path end → must have stopped
    }

    // upper_bound gives the first element > s_abs
    const auto it = std::upper_bound(cum.begin(), cum.end(), s_abs);
    const int idx =
        (int)(it - cum.begin());  // first index with cum[idx] > s_abs

    if (idx == 0)
    {
        return v_lim[0];  // before first waypoint
    }

    // Linear interpolation between waypoints [idx-1, idx]
    const double ds = cum[idx] - cum[idx - 1];
    if (ds < 1e-12)
    {
        return v_lim[idx];
    }
    const double t = (s_abs - cum[idx - 1]) / ds;
    return v_lim[idx - 1] * (1.0 - t) + v_lim[idx] * t;
}

/// Find the index of the segment whose arc-length interval contains s_abs.
/// Performs a binary search on cum[].  Result is clamped to [0, n_pts-2].
static int segmentAtArc(const std::vector<double>& cum, double s_abs)
{
    // upper_bound returns the first element > s_abs
    const auto it = std::upper_bound(cum.begin(), cum.end(), s_abs);
    int idx = (int)(it - cum.begin()) - 1;
    return std::clamp(idx, 0, (int)cum.size() - 2);
}

// ── distToEnd ─────────────────────────────────────────────────────────────────

double ReferenceGenerator::distToEnd(const Path& path, size_t idx, double t)
    const
{
    if (idx >= path.size() - 1)
    {
        return 0.0;
    }

    const double seg_len = (path.pts[idx + 1].pos - path.pts[idx].pos).norm();
    double d = (1.0 - t) * seg_len;

    for (size_t i = idx + 1; i < path.size() - 1; ++i)
    {
        d += (path.pts[i + 1].pos - path.pts[i].pos).norm();
    }

    return d;
}

// ── Main generator ────────────────────────────────────────────────────────────

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj,
    double v_cur) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    Reference r;
    r.x_ref.resize(N + 1);
    r.seg_normals.resize(N + 1);
    r.proj_pts.resize(N + 1);
    r.v_profile.resize(N + 1);

    const int n_pts = (int)path.size();

    // ── 1. Cumulative arc lengths ──────────────────────────────────────
    const std::vector<double> cum = cumulativeArcs(path);
    const double total = cum.back();

    // Segment lengths (n_pts-1 values)
    std::vector<double> segl(n_pts - 1);
    for (int i = 0; i < n_pts - 1; ++i)
    {
        segl[i] = cum[i + 1] - cum[i];
    }

    // ── 2. Arc position of the current projection ──────────────────────
    const size_t seg0 = proj.segment_index;
    const double s0 = cum[seg0] + proj.t * segl[seg0];

    // ── Edge case: robot already at or past path end ───────────────────
    if (s0 >= total - 1e-6)
    {
        const size_t last_seg = path.size() - 2;
        const Eigen::Vector2d end_pos = path.pts.back().pos;
        const Eigen::Vector2d end_dir = path.segmentDir(last_seg);
        const double end_theta = std::atan2(end_dir.y(), end_dir.x());
        const Eigen::Vector2d end_normal(-end_dir.y(), end_dir.x());

        for (int k = 0; k <= N; ++k)
        {
            r.x_ref[k] = {end_pos.x(), end_pos.y(), end_theta};
            r.seg_normals[k] = end_normal;
            r.proj_pts[k] = end_pos;
            r.v_profile[k] = 0.0;
        }
        return r;
    }

    // ── 3. Build backward-pass velocity profile over the whole path ───
    //
    // This precomputes, at every waypoint, the maximum speed from which the
    // robot can still brake to every upcoming corner and the path end in time.
    // The profile encodes all future speed constraints so that the forward
    // integration below can simply clamp against it without needing a per-step
    // scan over individual constraint events.
    //
    // This is mathematically equivalent to (and intentionally mirrors) the
    // backward-pass approach in control.py::_build_velocity_profile().
    std::vector<double> path_v_lim;
    buildVelocityProfile(
        path,
        cum,
        params_.v_max,
        params_.omega_max,
        params_.a_max,
        path_v_lim);

    // ── 4. Forward velocity integration ──────────────────────────────
    //
    // At each horizon step k:
    //  a) Look up the precomputed speed cap at the robot's current arc
    //     position (s0 + arc_at[k]).  Because the profile was built with a
    //     backward pass, this cap is the tightest constraint over ALL future
    //     events — not just the nearest — and accounts for cascading braking
    //     requirements between closely spaced corners.
    //  b) Clamp the velocity change to ±a_max·dt so the profile is
    //     kinematically continuous across MPC cycles.
    //
    // arc_at[k] = cumulative arc traversed by the velocity profile before
    //             step k (= distance from s0 to x_ref[k]).
    std::vector<double> arc_at(N + 1, 0.0);
    {
        double v = std::clamp(v_cur, 0.0, params_.v_max);

        for (int k = 0; k < N; ++k)
        {
            // Speed cap from the precomputed backward-pass profile.
            const double s_k = s0 + arc_at[k];
            const double v_cap =
                interpVelocityProfile(cum, path_v_lim, s_k, params_.v_max);

            // Advance velocity within ±a_max·dt (kinematic continuity).
            v = std::clamp(
                v_cap,
                v - params_.a_max * dt,
                v + params_.a_max * dt);
            v = std::max(v, 0.0);

            r.v_profile[k] = v;
            arc_at[k + 1] = arc_at[k] + v * dt;
        }

        // Terminal slot: repeat last velocity (consumed by the lineariser
        // for the final step but not used for arc advancement).
        r.v_profile[N] = (N > 0) ? r.v_profile[N - 1] : 0.0;
    }

    // ── 5. Sample reference positions at arc distances ─────────────────
    //
    // x_ref[k] is the desired state at time step k:
    //   • Position sampled from the polyline at s0 + arc_at[k].
    //   • Heading = direction of the polyline segment at that arc position.
    //
    // seg_normals[k] uses the EXPECTED PHYSICAL SEGMENT the robot will
    // occupy at step k (tracked via bisector advancement).  This separates
    // the lookahead reference geometry from the corridor constraint geometry,
    // which is essential for the QP to assign the correct half-space.

    size_t expected_robot_idx = proj.segment_index;
    Eigen::Vector2d expected_robot_pos = proj.proj;

    for (int k = 0; k <= N; ++k)
    {
        // ── Reference position ─────────────────────────────────────────
        const double s_abs = std::min(s0 + arc_at[k], total);

        const int seg_k = segmentAtArc(cum, s_abs);
        const double seg_len_k = segl[seg_k];
        const double t_k =
            (seg_len_k > 1e-12)
                ? std::clamp((s_abs - cum[seg_k]) / seg_len_k, 0.0, 1.0)
                : 0.0;

        const Eigen::Vector2d pos =
            path.pts[seg_k].pos +
            t_k * (path.pts[seg_k + 1].pos - path.pts[seg_k].pos);

        const Eigen::Vector2d ref_dir = path.segmentDir(seg_k);
        const double theta_ref = std::atan2(ref_dir.y(), ref_dir.x());

        // ── Physical corridor segment (bisector advancement) ───────────
        //
        // Advance expected_robot_idx while the EXPECTED robot position has
        // crossed the angle-bisector plane at the next junction.
        while (expected_robot_idx < path.size() - 2)
        {
            const Eigen::Vector2d A = path.pts[expected_robot_idx].pos;
            const Eigen::Vector2d B = path.pts[expected_robot_idx + 1].pos;
            const Eigen::Vector2d C = path.pts[expected_robot_idx + 2].pos;

            Eigen::Vector2d d1 = (B - A).normalized();
            Eigen::Vector2d d2 = (C - B).normalized();
            Eigen::Vector2d n_bisect = d1 + d2;
            if (n_bisect.squaredNorm() < 1e-6)
            {
                n_bisect = d1;
            }

            if ((expected_robot_pos - B).dot(n_bisect) > 0.0)
            {
                expected_robot_idx++;
            }
            else
            {
                break;
            }
        }

        const Eigen::Vector2d active_dir = path.segmentDir(expected_robot_idx);
        const Eigen::Vector2d physical_normal(-active_dir.y(), active_dir.x());

        // ── Store ──────────────────────────────────────────────────────
        r.x_ref[k] = {pos.x(), pos.y(), theta_ref};
        r.seg_normals[k] = physical_normal;

        // Project expected_robot_pos onto the active (physical) segment
        const Eigen::Vector2d pA = path.pts[expected_robot_idx].pos;
        const Eigen::Vector2d pB = path.pts[expected_robot_idx + 1].pos;
        const Eigen::Vector2d pAB = pB - pA;
        const double plen_sq = pAB.squaredNorm();
        Eigen::Vector2d corr_proj = pA;
        if (plen_sq > 1e-12)
        {
            const double t_corr = std::clamp(
                (expected_robot_pos - pA).dot(pAB) / plen_sq,
                0.0,
                1.0);
            corr_proj = pA + t_corr * pAB;
        }
        r.proj_pts[k] = corr_proj;  // physical projection, NOT pos (lookahead)

        // Advance expected robot position for the next step.
        if (k < N)
        {
            expected_robot_pos += active_dir * (r.v_profile[k] * dt);
        }
    }

    return r;
}

}  // namespace mpc
