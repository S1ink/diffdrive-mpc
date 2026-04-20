#pragma once

#include <vector>
#include <Eigen/Dense>

namespace mpc
{

struct PathPoint
{
    Eigen::Vector2d pos;
};

class Path
{
public:
    std::vector<PathPoint> pts;

    inline size_t size() const { return pts.size(); }

    Eigen::Vector2d segmentDir(size_t i) const
    {
        return (pts[i + 1].pos - pts[i].pos).normalized();
    }
};

}  // namespace mpc
