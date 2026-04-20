#pragma once

#include "types.hpp"
#include "path.hpp"
#include <limits>

namespace mpc
{

struct ProjectionResult
{
    size_t segment_index = 0;
    double t = 0.0;  // [0,1] parametric position along segment
    Eigen::Vector2d proj = Eigen::Vector2d::Zero();
};

/// Projects the robot onto the path.
///
/// Key properties vs. the original:
///   - Stateful: remembers the last matched segment so the search is always
///     forward-only (avoids re-snapping to a behind segment).
///   - Hysteresis: the tracked segment index only advances when t > seg_advance_t,
///     preventing rapid flipping near segment boundaries.
///   - Bounded look-ahead: searches at most `look_ahead` segments ahead of the
///     last matched one, so the cost is O(look_ahead) not O(path length).
class Projector
{
public:
    /// Only advance to the next segment when t exceeds this threshold.
    double seg_advance_t = 0.7;

    /// How many segments ahead of last_segment_ to search.
    size_t look_ahead = 5;

    /// Reset to the beginning of the path (call when the path changes
    /// drastically or the robot is re-localised far from the current position).
    void reset() { last_segment_ = 0; }

    ProjectionResult project(const State& x, const Path& path);

private:
    size_t last_segment_ = 0;
};

}  // namespace mpc
