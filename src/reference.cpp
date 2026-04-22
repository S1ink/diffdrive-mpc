#include "mpc/reference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mpc
{

// ── Internal types ────────────────────────────────────────────────────────────

/// A speed constraint at a specific arc-length position along the path.
struct VelocityEvent
{
    double s;      // arc-length position from path start [m]
    double v_lim;  // maximum speed at this point [m/s]
};

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

/// Find constraint events for the velocity profile:
///   • End-of-path stop:   (total_arc, 0)
///   • Each junction i where the heading change exceeds ~3°:
///         speed limit = ω_max × avg_segment_length / Δθ
///         (derived from ω = v·Δθ/ds_avg ≤ ω_max)
///
/// Only junctions ahead of s0 are included.
static std::vector<VelocityEvent> buildConstraintEvents(
    const Path& path,
    const std::vector<double>& cum,
    double s0,
    double v_max,
    double omega_max)
{
    std::vector<VelocityEvent> events;
    const int n = (int)path.size();

    // Always include an end-of-path stop.
    events.push_back({cum.back(), 0.0});

    for (int i = 1; i < n - 1; ++i)
    {
        const double s_j = cum[i];
        if (s_j <= s0 + 1e-6)
        {
            continue;  // behind or at current position
        }

        const Eigen::Vector2d d_in =
            (path.pts[i].pos - path.pts[i - 1].pos).normalized();
        const Eigen::Vector2d d_out =
            (path.pts[i + 1].pos - path.pts[i].pos).normalized();

        const double cos_a = std::clamp(d_in.dot(d_out), -1.0, 1.0);
        const double angle = std::acos(cos_a);  // [0, π]

        if (angle < 0.05)
        {
            continue;  // < ~3°, negligible curvature
        }

        // Average of incoming and outgoing segment lengths as arc estimate.
        const double l_in = (path.pts[i].pos - path.pts[i - 1].pos).norm();
        const double l_out = (path.pts[i + 1].pos - path.pts[i].pos).norm();
        const double ds_avg = 0.5 * (l_in + l_out);

        const double v_lim =
            std::clamp(omega_max * ds_avg / (angle + 1e-9), 0.0, v_max);

        events.push_back({s_j, v_lim});
    }

    return events;
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

    // ── 3. Build constraint events ────────────────────────────────────
    const std::vector<VelocityEvent> events =
        buildConstraintEvents(path, cum, s0, params_.v_max, params_.omega_max);

    // ── 4. Forward velocity integration with look-ahead braking ──────
    //
    // At each horizon step k the tightest allowable speed NOW is the minimum
    // over all upcoming events of:
    //
    //     v_cap = sqrt( v_ev² + 2·a_max·(s_ev − s_current) )
    //
    // This is the speed from which we can brake to v_ev by the time we
    // reach s_ev under constant deceleration a_max.  We then advance the
    // velocity within ± a_max·dt.
    //
    // arc_at[k] = cumulative arc traversed before step k
    //             (= distance from s0 to x_ref[k]).
    std::vector<double> arc_at(N + 1, 0.0);
    {
        double v = std::clamp(v_cur, 0.0, params_.v_max);

        for (int k = 0; k < N; ++k)
        {
            // Look-ahead: tightest speed cap over all future events.
            double v_cap = params_.v_max;
            for (const auto& ev : events)
            {
                // Remaining arc from current forward-integration position
                // to this event.
                const double ds = (ev.s - s0) - arc_at[k];
                if (ds >= 0.0)
                {
                    v_cap = std::min(
                        v_cap,
                        std::sqrt(
                            std::max(
                                0.0,
                                ev.v_lim * ev.v_lim +
                                    2.0 * params_.a_max * ds)));
                }
            }

            // Advance velocity within acceleration limits.
            v = std::clamp(
                v_cap,
                v - params_.a_max * dt,
                v + params_.a_max * dt);
            v = std::max(v, 0.0);

            r.v_profile[k] = v;
            arc_at[k + 1] = arc_at[k] + v * dt;
        }

        // Terminal slot: repeat last velocity (consumed by lineariser, not
        // used for advancing arc).
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
