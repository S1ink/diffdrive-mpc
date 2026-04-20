#include "mpc/reference.hpp"
#include <cmath>

namespace mpc
{

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj)
{
    Reference r;
    r.x_ref.resize(N);

    size_t idx = proj.segment_index;
    double t = proj.t;

    Eigen::Vector2d pos = proj.proj;

    for (int k = 0; k < N; ++k)
    {
        if (idx >= path.size() - 1)
        {
            idx = path.size() - 2;
        }

        auto A = path.pts[idx].pos;
        auto B = path.pts[idx + 1].pos;

        Eigen::Vector2d dir = (B - A).normalized();
        pos += dir * v_ref * dt;

        if ((pos - B).dot(dir) > 0.0)
        {
            idx++;
        }

        double theta = std::atan2(dir.y(), dir.x());

        r.x_ref[k] = {pos.x(), pos.y(), theta};
    }

    return r;
}

}  // namespace mpc
