#include "mpc/reference.hpp"

#include <cmath>
#include <algorithm>

namespace mpc
{

// ── Helpers ───────────────────────────────────────────────────────────────────

double ReferenceGenerator::curvatureAt(const Path& path, size_t idx) const
{
    // Curvature at the end of segment `idx` (i.e. the upcoming turn).
    // Returns 0 if there is no next segment.
    if (idx + 2 >= path.size())
    {
        return 0.0;
    }

    const Eigen::Vector2d d1 =
        (path.pts[idx + 1].pos - path.pts[idx].pos).normalized();
    const Eigen::Vector2d d2 =
        (path.pts[idx + 2].pos - path.pts[idx + 1].pos).normalized();

    // Interior angle between the two segments
    const double cos_a = std::clamp(d1.dot(d2), -1.0, 1.0);
    const double angle = std::acos(cos_a);  // [0, π]

    // Arc-length estimate: average of the two segment lengths
    const double l1 = (path.pts[idx + 1].pos - path.pts[idx].pos).norm();
    const double l2 = (path.pts[idx + 2].pos - path.pts[idx + 1].pos).norm();
    const double arc = 0.5 * (l1 + l2) + 1e-9;

    return angle / arc;
}

double ReferenceGenerator::distToEnd(const Path& path, size_t idx, double t)
    const
{
    if (idx >= path.size() - 1)
    {
        return 0.0;
    }

    // Remaining portion of the current segment
    const double seg_len = (path.pts[idx + 1].pos - path.pts[idx].pos).norm();
    double d = (1.0 - t) * seg_len;

    // Full subsequent segments
    for (size_t i = idx + 1; i < path.size() - 1; ++i)
    {
        d += (path.pts[i + 1].pos - path.pts[i].pos).norm();
    }

    return d;
}

// ── Main generator ────────────────────────────────────────────────────────────

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    Reference r;
    r.x_ref.resize(N + 1);
    r.seg_normals.resize(N + 1);
    r.proj_pts.resize(N + 1);
    r.v_profile.resize(N + 1);

    size_t idx = proj.segment_index;
    double t = proj.t;
    Eigen::Vector2d pos = proj.proj;
    Eigen::Vector2d expected_robot_pos(proj.proj.x(), proj.proj.y());
    size_t expected_robot_idx = proj.segment_index;

    for (int k = 0; k <= N; ++k)
    {
        // Clamp to last valid segment at path end
        if (idx >= path.size() - 1)
        {
            idx = path.size() - 2;
        }

        const Eigen::Vector2d A = path.pts[idx].pos;
        const Eigen::Vector2d B = path.pts[idx + 1].pos;
        const Eigen::Vector2d dir = (B - A).normalized();

        // Unit normal: left of the travel direction
        const Eigen::Vector2d normal(-dir.y(), dir.x());

        // ── Velocity profile ──────────────────────────────────────────
        // v_ref_k = min( v_max,  v_cruise,
        //               ω_max / (κ + ε),           ← curvature limit
        //               sqrt(2·a_max·d_remaining)   ← braking limit )
        const double kappa = curvatureAt(path, idx);
        const double d_end = distToEnd(path, idx, t);
        const double v_curvature = params_.omega_max / (kappa + 1e-6);
        const double v_stop = std::sqrt(2.0 * params_.a_max * (d_end + 1e-6));
        const double v_k =
            std::min({params_.v_max, params_.v_ref, v_curvature, v_stop});

        // ── Bisector Update for Corridor Assignment ───────────────────
        while (expected_robot_idx < path.size() - 2)
        {
            const Eigen::Vector2d A = path.pts[expected_robot_idx].pos;
            const Eigen::Vector2d B = path.pts[expected_robot_idx + 1].pos;
            const Eigen::Vector2d C = path.pts[expected_robot_idx + 2].pos;

            Eigen::Vector2d d1 = (B - A).normalized();
            Eigen::Vector2d d2 = (C - B).normalized();
            Eigen::Vector2d n_bisect = d1 + d2;
            if (n_bisect.squaredNorm() < 1e-6) n_bisect = d1;

            if ((expected_robot_pos - B).dot(n_bisect) > 0.0) {
                expected_robot_idx++;
            } else {
                break;
            }
        }

        // Assign the normal based on the EXPECTED physical segment, not the reference segment
        const Eigen::Vector2d active_dir = path.segmentDir(expected_robot_idx);
        const Eigen::Vector2d physical_normal(-active_dir.y(), active_dir.x());

        // Store reference for this horizon step
        r.x_ref[k] = {pos.x(), pos.y(), std::atan2(path.segmentDir(idx).y(), path.segmentDir(idx).x())};
        r.seg_normals[k] = physical_normal; 
        r.proj_pts[k] = pos;
        r.v_profile[k] = v_k;

        // ── Advance reference position along path ─────────────────────
        // Walk exactly v_k * dt metres forward, crossing segment
        // boundaries as needed.
        double step = v_k * dt;

        while (step > 1e-9)
        {
            if (idx >= path.size() - 1)
            {
                // Already at the end — stay put
                pos = path.pts.back().pos;
                step = 0.0;
                break;
            }

            const Eigen::Vector2d Ai = path.pts[idx].pos;
            const Eigen::Vector2d Bi = path.pts[idx + 1].pos;
            const double seg_len = (Bi - Ai).norm();
            const double remaining = (1.0 - t) * seg_len;

            if (step < remaining)
            {
                t += step / (seg_len + 1e-12);
                pos = Ai + t * (Bi - Ai);
                step = 0.0;
            }
            else
            {
                step -= remaining;
                idx++;
                t = 0.0;
                pos = path.pts[idx].pos;
            }
        }

        expected_robot_pos += active_dir * (v_k * dt);
    }

    return r;
}

}  // namespace mpc
