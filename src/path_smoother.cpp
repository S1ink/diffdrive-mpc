#include "mpc/path_smoother.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>


namespace mpc
{

// File-scope geometry helpers (mirror of PathSampler helpers)

namespace
{

// tan(theta/2)
static inline double halfTan(double theta) { return std::tan(theta * 0.5); }

// cos(theta/2) / (1 - cos(theta/2))
// Used to derive an initial radius proportional to the junction sharpness.
static inline double rho(double theta)
{
    const double a = std::cos(theta * 0.5);
    return a / (1.0 - a + 1e-12);
}

// Wrap angle to [-pi, pi]
static inline double wrapAngle(double a)
{
    constexpr double PI = std::numbers::pi;
    constexpr double PI2 = PI * 2.0;
    while (a > PI)
    {
        a -= PI2;
    }
    while (a < -PI)
    {
        a += PI2;
    }
    return a;
}

}  // namespace

// SmoothedPath helpers

int PathSmoother::SmoothedPath::segAt(double s) const
{
    if (segs.empty())
    {
        return 0;
    }
    // upper_bound -> first cum > s -> segment index = that - 1
    const auto it = std::upper_bound(cum.begin(), cum.end(), s);
    const int idx = static_cast<int>(it - cum.begin()) - 1;
    return std::clamp(idx, 0, static_cast<int>(segs.size()) - 1);
}

// project

std::pair<double, Eigen::Vector2d> PathSmoother::SmoothedPath::project(
    const Eigen::Vector2d& p) const
{
    if (segs.empty())
    {
        return {0.0, p};
    }

    constexpr double PI2 = std::numbers::pi * 2.0;

    double best_s = 0.0;
    double best_d2 = std::numeric_limits<double>::max();
    Eigen::Vector2d best_pt = p;

    for (int i = 0; i < static_cast<int>(segs.size()); ++i)
    {
        const double s_start = cum[i];

        if (std::holds_alternative<LineSegment>(segs[i]))
        {
            const auto& seg = std::get<LineSegment>(segs[i]);
            if (seg.length < 1e-12)
            {
                continue;
            }

            const double t =
                std::clamp((p - seg.start).dot(seg.dir), 0.0, seg.length);
            const Eigen::Vector2d pt = seg.start + t * seg.dir;
            const double d2 = (pt - p).squaredNorm();

            if (d2 < best_d2)
            {
                best_d2 = d2;
                best_pt = pt;
                best_s = s_start + t;
            }
        }
        else
        {
            const auto& seg = std::get<ArcSegment>(segs[i]);
            const bool ccw = (seg.sweep_angle > 0.0);
            const double ea = seg.endAngle();
            const double lo = std::min(seg.start_angle, ea);
            const double hi = std::max(seg.start_angle, ea);

            // Candidate angle: direction from center to p
            double theta =
                std::atan2(p.y() - seg.center.y(), p.x() - seg.center.x());

            // Try +/- 2*pi aliases so we can land inside [lo, hi]
            if (!(lo <= theta && theta <= hi))
            {
                const double tp = theta + PI2;
                const double tm = theta - PI2;
                if (lo <= tp && tp <= hi)
                {
                    theta = tp;
                }
                else if (lo <= tm && tm <= hi)
                {
                    theta = tm;
                }
                else
                {
                    // Clamp to nearer endpoint
                    const double ds =
                        std::abs(wrapAngle(theta - seg.start_angle));
                    const double de = std::abs(wrapAngle(theta - ea));
                    theta = (ds < de) ? seg.start_angle : ea;
                }
            }
            theta = std::clamp(theta, lo, hi);

            const Eigen::Vector2d pt =
                seg.center +
                Eigen::Vector2d{std::cos(theta), std::sin(theta)} * seg.radius;
            const double d2 = (pt - p).squaredNorm();

            if (d2 < best_d2)
            {
                best_d2 = d2;
                best_pt = pt;

                // Arc distance from start_angle to theta along the arc's direction
                const double dtheta =
                    ccw ? (theta - seg.start_angle) : (seg.start_angle - theta);
                best_s = s_start + std::max(0.0, dtheta) * seg.radius;
            }
        }
    }

    return {best_s, best_pt};
}

// sampleAt

PathSmoother::SmoothSample PathSmoother::SmoothedPath::sampleAt(
    double s,
    double v_max_global) const
{
    if (segs.empty())
    {
        return {
            Eigen::Vector2d::Zero(),
            0.0,
            Eigen::Vector2d{0.0, 1.0},
            v_max_global
        };
    }

    s = std::clamp(s, 0.0, total);

    constexpr double PIH = std::numbers::pi * 0.5;

    const int i = segAt(s);
    const double ds = s - cum[i];

    SmoothSample out;

    if (std::holds_alternative<LineSegment>(segs[i]))
    {
        const auto& seg = std::get<LineSegment>(segs[i]);
        const double t = std::min(ds, seg.length);

        out.pos = seg.start + t * seg.dir;
        out.heading = std::atan2(seg.dir.y(), seg.dir.x());
        out.normal = Eigen::Vector2d{-seg.dir.y(), seg.dir.x()};
        out.v_limit = v_max_global;
    }
    else
    {
        const auto& seg = std::get<ArcSegment>(segs[i]);
        const bool ccw = (seg.sweep_angle > 0.0);

        // Angular progress along the arc
        const double max_sweep = std::abs(seg.sweep_angle);
        const double dtheta = std::clamp(ds / seg.radius, 0.0, max_sweep);
        const double theta = seg.start_angle + (ccw ? dtheta : -dtheta);
        const double theta_tan = theta + (ccw ? PIH : -PIH);

        const Eigen::Vector2d tan_dir{std::cos(theta_tan), std::sin(theta_tan)};

        out.pos =
            seg.center +
            Eigen::Vector2d{std::cos(theta), std::sin(theta)} * seg.radius;
        out.heading = wrapAngle(theta_tan);
        out.normal = Eigen::Vector2d{-tan_dir.y(), tan_dir.x()};
        out.v_limit = seg.v_max;
    }

    return out;
}

// PathSmoother::buildJunctions

void PathSmoother::buildJunctions(
    const Path& path,
    std::vector<Junction>& juncs,
    std::vector<double>& seg_lengths,
    std::vector<double>& half_tans) const
{
    const int n_pts = static_cast<int>(path.size());
    const int n_segs = n_pts - 1;
    const int n_juncs = n_pts - 2;

    assert(n_juncs >= 1);

    // Segment lengths
    seg_lengths.resize(n_segs);
    for (int i = 0; i < n_segs; ++i)
    {
        seg_lengths[i] = (path.pts[static_cast<size_t>(i + 1)].pos -
                          path.pts[static_cast<size_t>(i)].pos)
                             .norm();
    }

    // Radius caps: kinematic cap r_kin = v_max / omega_max and geometric
    // cap r_corr = d_hard * rho(theta). The chosen radius is the tighter
    // of these bounds.
    const double r_kin = (params_.omega_max > 1e-9)
                             ? params_.v_max / params_.omega_max
                             : std::numeric_limits<double>::max();

    // Initial junction radii
    half_tans.resize(n_juncs);
    juncs.resize(n_juncs);

    for (int j = 0; j < n_juncs; ++j)
    {
        const Eigen::Vector2d s_in = (path.pts[static_cast<size_t>(j + 1)].pos -
                                      path.pts[static_cast<size_t>(j)].pos)
                                         .normalized();
        const Eigen::Vector2d s_out =
            (path.pts[static_cast<size_t>(j + 2)].pos -
             path.pts[static_cast<size_t>(j + 1)].pos)
                .normalized();

        const double cos_a = std::clamp(s_in.dot(s_out), -1.0, 1.0);
        juncs[j].theta = std::acos(cos_a);  // exterior turn angle in [0, pi]
        half_tans[j] = halfTan(juncs[j].theta);

        // Corridor deviation cap: r <= d_hard * rho(theta)
        // rho(theta) is the same helper used by PathSampler for the same purpose.
        // For theta near pi (U-turn), rho -> 0 and r_corr -> 0, which is correct:
        // a near-180-degree turn cannot be smoothed without leaving the corridor,
        // so no arc is inserted and the robot must turn on the spot.
        const double r_corr = params_.d_hard * rho(juncs[j].theta);

        // Initial radius: tightest of kinematic and corridor caps, then
        // additionally bounded by the first/last segment lengths (the
        // multi-pass sweep handles all interior segments).
        double r = std::min(r_kin, r_corr);
        if (j == 0 && half_tans[j] > 1e-9)
        {
            r = std::min(r, seg_lengths[0] / half_tans[j]);
        }
        if (j == n_juncs - 1 && half_tans[j] > 1e-9)
        {
            r = std::min(r, seg_lengths[n_segs - 1] / half_tans[j]);
        }

        juncs[j].radius = r;
    }

    // Multi-pass overlap resolution (forward + backward sweeps).
    constexpr int kSweepPairs = 5;
    for (int sweep = 0; sweep < kSweepPairs; ++sweep)
    {
        for (int s = 1; s < n_segs - 1; ++s)
        {
            optimizeJunction(
                static_cast<size_t>(s),
                juncs,
                seg_lengths,
                half_tans);
        }
        for (int s = n_segs - 2; s > 0; --s)
        {
            optimizeJunction(
                static_cast<size_t>(s),
                juncs,
                seg_lengths,
                half_tans);
        }
    }

    // Finalise: compute derived quantities
    for (int j = 0; j < n_juncs; ++j)
    {
        juncs[j].tan_off = juncs[j].radius * half_tans[j];
        juncs[j].v_max =
            std::min(params_.v_max, juncs[j].radius * params_.omega_max);
    }
}

// PathSmoother::optimizeJunction

void PathSmoother::optimizeJunction(
    size_t seg_i,
    std::vector<Junction>& juncs,
    const std::vector<double>& seg_lengths,
    const std::vector<double>& half_tans) const
{
    // seg_i is the SEGMENT index (1-based interior).
    // The two junctions bordering this segment are j_left = seg_i-1 and
    // j_right = seg_i.
    const size_t j_left = seg_i - 1;
    const size_t j_right = seg_i;

    Junction& jn_l = juncs[j_left];
    Junction& jn_r = juncs[j_right];

    const double ht_l = half_tans[j_left];
    const double ht_r = half_tans[j_right];
    const double budget = seg_lengths[seg_i];
    const double tan_l = jn_l.radius * ht_l;
    const double tan_r = jn_r.radius * ht_r;

    if (tan_l + tan_r <= budget)
    {
        return;  // already fits
    }

    const double half = budget * 0.5;

    if (tan_l <= half)
    {
        // Left fits in its half; right gets the remainder.
        if (ht_r > 1e-9)
        {
            jn_r.radius = std::min(jn_r.radius, (budget - tan_l) / ht_r);
        }
    }
    else if (tan_r <= half)
    {
        // Right fits in its half; left gets the remainder.
        if (ht_l > 1e-9)
        {
            jn_l.radius = std::min(jn_l.radius, (budget - tan_r) / ht_l);
        }
    }
    else
    {
        // Both exceed their half; split evenly.
        if (ht_l > 1e-9)
        {
            jn_l.radius = std::min(jn_l.radius, half / ht_l);
        }
        if (ht_r > 1e-9)
        {
            jn_r.radius = std::min(jn_r.radius, half / ht_r);
        }
    }
}

// PathSmoother::smooth

PathSmoother::SmoothedPath PathSmoother::smooth(const Path& path) const
{
    SmoothedPath sp;

    if (path.size() < 2)
    {
        return sp;  // empty
    }

    // Trivial case: two-point path = single line
    if (path.size() == 2)
    {
        const Eigen::Vector2d start = path.pts[0].pos;
        const Eigen::Vector2d end = path.pts[1].pos;
        const double length = (end - start).norm();
        if (length < 1e-12)
        {
            return sp;
        }

        LineSegment seg;
        seg.start = start;
        seg.end = end;
        seg.dir = (end - start) / length;
        seg.length = length;

        sp.cum.push_back(0.0);
        sp.segs.push_back(seg);
        sp.total = length;
        return sp;
    }

    // General case: build junctions
    std::vector<Junction> juncs;
    std::vector<double> seg_lengths, half_tans;
    buildJunctions(path, juncs, seg_lengths, half_tans);

    const int n_pts = static_cast<int>(path.size());
    const int n_juncs = n_pts - 2;

    sp.segs.reserve(static_cast<size_t>(2 * n_pts));
    sp.cum.reserve(static_cast<size_t>(2 * n_pts));

    // Lambda: append a LineSegment from a to b (skips degenerate segments).
    auto push_line = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b)
    {
        const double len = (b - a).norm();
        if (len < 1e-9)
        {
            return;
        }

        LineSegment seg;
        seg.start = a;
        seg.end = b;
        seg.dir = (b - a) / len;
        seg.length = len;

        sp.cum.push_back(sp.total);
        sp.segs.push_back(seg);
        sp.total += len;
    };

    // prev_end tracks where the previous segment ended.  Starts at the
    // path origin and advances as line segments and arc exits are appended.
    Eigen::Vector2d prev_end = path.pts.front().pos;

    for (int i = 0; i < n_juncs; ++i)
    {
        const Eigen::Vector2d& prev_pt = path.pts[static_cast<size_t>(i)].pos;
        const Eigen::Vector2d& curr_pt =
            path.pts[static_cast<size_t>(i + 1)].pos;
        const Eigen::Vector2d& next_pt =
            path.pts[static_cast<size_t>(i + 2)].pos;
        const Junction& jn = juncs[static_cast<size_t>(i)];

        // Degenerate: near-straight junction -> straight through
        if (jn.radius < 1e-9 || jn.theta < 1e-4)
        {
            push_line(prev_end, curr_pt);
            prev_end = curr_pt;
        }
        else
        {
            // Tangent directions
            const Eigen::Vector2d s1 = (curr_pt - prev_pt).normalized();
            const Eigen::Vector2d s2 = (next_pt - curr_pt).normalized();

            // Arc entry: curr_pt pulled back along s1 by the tangent offset.
            Eigen::Vector2d arc_beg = curr_pt - s1 * jn.tan_off;

            // Guard: floating-point can push arc_beg behind prev_end for
            // nearly-saturated adjacent arcs.  Snap forward if needed.
            if (s1.dot(arc_beg - prev_end) < 0.0)
            {
                arc_beg = prev_end;
            }

            push_line(prev_end, arc_beg);

            // Arc center (perpendicular to s1, toward inside of turn)
            // 2-D cross product sign: s1 x s2 > 0 -> center is on the left
            const bool center_left = (s1.x() * s2.y() > s1.y() * s2.x());
            const Eigen::Vector2d to_center =
                center_left ? Eigen::Vector2d{-s1.y(), s1.x()}
                            : Eigen::Vector2d{s1.y(), -s1.x()};

            ArcSegment arc;
            arc.center = arc_beg + to_center * jn.radius;
            arc.radius = jn.radius;
            arc.start_angle = std::atan2(
                arc_beg.y() - arc.center.y(),
                arc_beg.x() - arc.center.x());
            arc.sweep_angle = center_left ? jn.theta : -jn.theta;
            arc.v_max = jn.v_max;

            sp.cum.push_back(sp.total);
            sp.segs.push_back(arc);
            sp.total += arc.arcLength();

            // Arc exit: the symmetric tangent point on the outgoing segment
            prev_end = curr_pt + s2 * jn.tan_off;
        }

        // After the last junction, close the path to the final waypoint.
        if (i + 1 == n_juncs)
        {
            push_line(prev_end, path.pts.back().pos);
        }
    }

    return sp;
}

}  // namespace mpc
