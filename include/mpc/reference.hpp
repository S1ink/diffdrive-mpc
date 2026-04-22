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
};

class ReferenceGenerator
{
public:
    explicit ReferenceGenerator(const MPCParams& p) : params_(p) {}

    /// Build a horizon-length reference starting from the current projection.
    ///
    /// @param path   Current path polyline.
    /// @param proj   Projection result for the robot's current (latency-
    ///               compensated) position.
    /// @param v_cur  Current robot forward speed [m/s].  Passed as the seed
    ///               for the forward velocity integration so the profile is
    ///               continuous across MPC cycles.  Default 0 (cold start).
    Reference generate(
        const Path& path,
        const ProjectionResult& proj,
        double v_cur = 0.0) const;

private:
    MPCParams params_;

    // ── Internal helpers ──────────────────────────────────────────────

    /// Arc length remaining from (idx, t) to the end of the path [m].
    double distToEnd(const Path& path, size_t idx, double t) const;
};

}  // namespace mpc