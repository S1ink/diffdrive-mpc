#pragma once

// =============================================================================
// path_smoother.hpp — Polyline-to-smooth-geometry converter for the MPC
//                     reference generator.
//
// At each interior waypoint of a polyline path, PathSmoother replaces the
// sharp corner with a circular arc whose radius satisfies two constraints:
//
//   1. Kinematic:  radius ≤ v_max / omega_max
//      (ensures the robot can traverse the arc without exceeding ω_max)
//
//   2. Geometric:  the tangent offset (radius · tan(θ/2)) must fit inside
//      the adjacent segment lengths without the arcs of two neighbouring
//      junctions overlapping.  An iterative forward–backward sweep (ported
//      from the Stanley-controller PathSampler) enforces this globally.
//
// The resulting SmoothedPath supports:
//   • Projection of a 2-D point → (arc position, closest point)
//   • Sampling (position, tangent heading, left-normal, speed limit) at
//     any arc position along the smooth path.
//
// These two operations replace the raw-polyline arc-length table used by the
// original ReferenceGenerator so that reference headings are kinematically
// continuous through corners.
// =============================================================================

#include "path.hpp"
#include "params.hpp"

#include <Eigen/Dense>
#include <limits>
#include <numbers>
#include <variant>
#include <vector>

namespace mpc
{

class PathSmoother
{
public:
    // ── Segment types ─────────────────────────────────────────────────────────

    struct LineSegment
    {
        Eigen::Vector2d start{Eigen::Vector2d::Zero()};
        Eigen::Vector2d end{Eigen::Vector2d::Zero()};
        Eigen::Vector2d dir{Eigen::Vector2d::Zero()};  // unit tangent
        double length{0.0};
    };

    struct ArcSegment
    {
        Eigen::Vector2d center{Eigen::Vector2d::Zero()};
        double radius{0.0};
        double start_angle{0.0};
        double sweep_angle{0.0};  // positive = CCW, negative = CW
        double v_max{0.0};        // speed cap on this arc [m/s]

        double arcLength() const { return radius * std::abs(sweep_angle); }
        double endAngle() const { return start_angle + sweep_angle; }
    };

    using Segment = std::variant<LineSegment, ArcSegment>;

    // ── Per-point sample from the smooth path ─────────────────────────────────

    struct SmoothSample
    {
        Eigen::Vector2d pos{Eigen::Vector2d::Zero()};
        double heading{0.0};  // tangent direction [rad]
        Eigen::Vector2d normal{
            Eigen::Vector2d::Zero()};  // left normal to tangent
        double v_limit{0.0};           // speed cap at this point
    };

    // ── Smooth path container ─────────────────────────────────────────────────

    struct SmoothedPath
    {
        std::vector<Segment> segs;
        std::vector<double>
            cum;            // cum[i] = arc length from path start to segs[i]
        double total{0.0};  // total arc length [m]

        bool empty() const { return segs.empty(); }

        /// Project a 2-D point p onto the smooth path.
        /// Returns {arc_position_s, closest_point_on_path}.
        std::pair<double, Eigen::Vector2d> project(
            const Eigen::Vector2d& p) const;

        /// Sample position, tangent heading, left-normal, and speed limit at
        /// arc position s (clamped to [0, total]).
        /// v_max_global is returned on line segments where no arc limit applies.
        SmoothSample sampleAt(double s, double v_max_global) const;

    private:
        /// Binary-search helper: index of the segment that contains arc pos s.
        int segAt(double s) const;
    };

    // ── Public API ─────────────────────────────────────────────────────────────

    explicit PathSmoother(const MPCParams& p) : params_(p) {}

    /// Convert a raw polyline path to a smooth line+arc sequence.
    SmoothedPath smooth(const Path& path) const;

private:
    MPCParams params_;

    // ── Junction metadata ─────────────────────────────────────────────────────

    struct Junction
    {
        double theta{0.0};    // exterior turn angle [0, π]
        double radius{0.0};   // fitted arc radius [m]
        double tan_off{0.0};  // radius · tan(θ/2): segment length consumed
        double v_max{0.0};    // speed limit on the arc
    };

    // ── Internal helpers ───────────────────────────────────────────────────────

    /// Compute per-junction radii from kinematic and geometric constraints,
    /// then run the multi-pass optimisation sweep to prevent arc overlap.
    void buildJunctions(
        const Path& path,
        std::vector<Junction>& juncs,
        std::vector<double>& seg_lengths,
        std::vector<double>& half_tans) const;

    /// Single-segment overlap resolution step (called by buildJunctions).
    void optimizeJunction(
        size_t seg_i,
        std::vector<Junction>& juncs,
        const std::vector<double>& seg_lengths,
        const std::vector<double>& half_tans) const;
};

}  // namespace mpc
