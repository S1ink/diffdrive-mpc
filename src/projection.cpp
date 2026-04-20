#include "mpc/projection.hpp"

namespace mpc
{

ProjectionResult Projector::project(const State& x, const Path& path)
{
    ProjectionResult best;
    double best_dist = 1e9;

    Eigen::Vector2d p(x.x, x.y);

    for (size_t i = 0; i < path.size() - 1; ++i)
    {
        auto A = path.pts[i].pos;
        auto B = path.pts[i + 1].pos;

        Eigen::Vector2d AB = B - A;
        double t = (p - A).dot(AB) / AB.squaredNorm();
        t = std::clamp(t, 0.0, 1.0);

        Eigen::Vector2d proj = A + t * AB;
        double d = (p - proj).norm();

        if (d < best_dist)
        {
            best_dist = d;
            best.segment_index = i;
            best.t = t;
            best.proj = proj;
        }
    }

    return best;
}

}  // namespace mpc
