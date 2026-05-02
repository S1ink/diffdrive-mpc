#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
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

    bool valid() const { return pts.size() >= 2; }

    Eigen::Vector2d segmentDir(size_t i) const
    {
        assert(i + 1 < pts.size());
        return (pts[i + 1].pos - pts[i].pos).normalized();
    }
};

/// Cumulative arc-lengths along a path.
/// Returns a vector of length path.size() where cum[i] is the arc-length
/// from pts[0] to pts[i].  cum[0] = 0 always.
inline std::vector<double> cumulativeArcs(const Path& path)
{
    const int n = (int)path.size();
    std::vector<double> cum(n, 0.0);
    for (int i = 1; i < n; ++i)
    {
        cum[i] = cum[i - 1] + (path.pts[i].pos - path.pts[i - 1].pos).norm();
    }
    return cum;
}

/// Binary-search for the segment index whose arc-length interval contains s_abs.
/// Result is always clamped to [0, cum.size()-2].
inline int segmentAtArc(const std::vector<double>& cum, double s_abs)
{
    const auto it = std::upper_bound(cum.begin(), cum.end(), s_abs);
    int idx = (int)(it - cum.begin()) - 1;
    return std::clamp(idx, 0, (int)cum.size() - 2);
}

}  // namespace mpc
