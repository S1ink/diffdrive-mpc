#pragma once

#include "types.hpp"
#include "path.hpp"
#include "projection.hpp"
#include "params.hpp"

#include <vector>
#include <Eigen/Dense>

namespace mpc
{

/// Everything the QPBuilder needs for one horizon.
/// All vectors have length N+1 (indexed k = 0 .. N).
struct Reference
{
    std::vector<State> x_ref;  // reference state  [x, y, θ]
    std::vector<Eigen::Vector2d>
        seg_normals;                        // unit normal to assigned segment
    std::vector<Eigen::Vector2d> proj_pts;  // corresponding point on the path
    std::vector<double> v_profile;          // adaptive reference speed [m/s]
    double cte{0.0};  // signed cross-track error from smooth path [m]
};

class ReferenceGenerator
{
public:
    explicit ReferenceGenerator(const MPCParams& p) : params_(p) {}

    /// Build a horizon-length reference starting from the current projection.
    ///
    /// @param path       Current path polyline.
    /// @param proj       Projection result for the robot's current (latency-
    ///                   compensated) position onto the raw polyline.
    ///                   Used for seg_normals / proj_pts corridor geometry and
    ///                   for the velocity-event search window.
    /// @param robot_pos  Latency-compensated robot 2-D position [m].
    ///                   Projected directly onto the smooth path to obtain the
    ///                   arc-position seed s0.  Using the robot position rather
    ///                   than proj.proj eliminates the s0 discontinuity that
    ///                   occurred when the raw Projector crossed the bisector
    ///                   plane at a corner and proj.proj jumped from one raw
    ///                   segment to the next.
    /// @param v_cur      Current robot forward speed [m/s].  Passed as the seed
    ///                   for the forward velocity integration so the profile is
    ///                   continuous across MPC cycles.  Default 0 (cold start).
    Reference generate(
        const Path& path,
        const ProjectionResult& proj,
        const Eigen::Vector2d& robot_pos,
        double v_cur = 0.0) const;

private:
    MPCParams params_;

    // ── Internal helpers ──────────────────────────────────────────────

    /// Arc length remaining from (idx, t) to the end of the path [m].
    double distToEnd(const Path& path, size_t idx, double t) const;
};

}  // namespace mpc