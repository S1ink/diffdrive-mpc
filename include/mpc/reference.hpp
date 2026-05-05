#pragma once

#include "path.hpp"
#include "types.hpp"
#include "params.hpp"
#include "path_smoother.hpp"

#include <Eigen/Dense>

#include <vector>


namespace mpc
{

// Everything the QPBuilder needs for one horizon.
// All vectors have length N+1 (indexed k = 0 .. N).
struct Reference
{
    std::vector<State> x_ref;  // reference state  [x, y, theta]
    std::vector<Eigen::Vector2d>
        seg_normals;                        // unit normal to assigned segment
    std::vector<Eigen::Vector2d> proj_pts;  // corresponding point on the path
    std::vector<double> v_profile;          // adaptive reference speed [m/s]
    double cte{0.0};  // signed cross-track error from smooth path [m]
    double remaining_arc{
        0.0};  // arc length from robot projection to path end [m]
};

class ReferenceGenerator
{
public:
    explicit ReferenceGenerator(const MPCParams& p) : params_(p), smoother_(p)
    {
    }

    // Build a horizon-length reference starting from the current projection.
    //
    // @param path       Current path polyline.
    // @param path_hash  Hash of path geometry (from MPCController::hashPath).
    //                   Used to decide whether to rebuild the cached SmoothedPath.
    // @param robot_pos  Latency-compensated robot 2-D position [m].
    //                   Projected directly onto the smooth path to obtain s0.
    // @param v_cur      Current robot forward speed [m/s].
    Reference generate(
        const Path& path,
        size_t path_hash,
        const Eigen::Vector2d& robot_pos,
        double v_cur = 0.0);

private:
    MPCParams params_;
    PathSmoother smoother_;

    size_t cached_hash_{0};
    PathSmoother::SmoothedPath cached_sp_;
    bool has_cached_sp_{false};
};

}  // namespace mpc
