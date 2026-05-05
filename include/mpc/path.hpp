#pragma once

#include <Eigen/Dense>

#include <vector>


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

    bool valid() const { return pts.size() >= 2; }

    Eigen::Vector2d segmentDir(size_t i) const
    {
        assert(i + 1 < pts.size());
        return (pts[i + 1].pos - pts[i].pos).normalized();
    }
};

}  // namespace mpc
