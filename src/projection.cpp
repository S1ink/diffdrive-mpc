#include "mpc/projection.hpp"

#include <stdexcept>


namespace mpc
{

ProjectionResult Projector::project(const State& x, const Path& path)
{
    if (!path.valid())
    {
        throw std::invalid_argument("Path must have >= 2 points");
    }

    const Eigen::Vector2d p(x.x, x.y);

    // Clamp starting segment to valid range
    if (last_segment_ >= path.size() - 1)
    {
        last_segment_ = path.size() - 2;
    }

    // Angle-bisector advancement: advance while the robot crosses the
    // bisector plane of the next waypoint.
    while (last_segment_ < path.size() - 2)
    {
        const Eigen::Vector2d A = path.pts[last_segment_].pos;
        const Eigen::Vector2d B = path.pts[last_segment_ + 1].pos;
        const Eigen::Vector2d C = path.pts[last_segment_ + 2].pos;

        const Eigen::Vector2d dir1 = (B - A).normalized();
        const Eigen::Vector2d dir2 = (C - B).normalized();

        // The normal of the bisector plane (pointing "forward" into the next segment)
        // is simply the sum of the two unit direction vectors.
        Eigen::Vector2d n_bisect = dir1 + dir2;

        // Fallback if the path doubles back on itself perfectly (U-turn)
        if (n_bisect.squaredNorm() < 1e-6)
        {
            n_bisect = dir1;
        }

        // If the dot product is positive, point 'p' has crossed the bisector plane
        if ((p - B).dot(n_bisect) > 0.0)
        {
            last_segment_++;
        }
        else
        {
            break;  // Still in the region of the current segment
        }
    }

    // Final projection onto the active segment
    const Eigen::Vector2d A = path.pts[last_segment_].pos;
    const Eigen::Vector2d B = path.pts[last_segment_ + 1].pos;
    const Eigen::Vector2d AB = B - A;
    const double len_sq = AB.squaredNorm();

    ProjectionResult best;
    best.segment_index = last_segment_;
    best.t = 0.0;

    if (len_sq > 1e-12)
    {
        best.t = std::clamp((p - A).dot(AB) / len_sq, 0.0, 1.0);
    }

    best.proj = A + best.t * AB;
    return best;
}

}  // namespace mpc
