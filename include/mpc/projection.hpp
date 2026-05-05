#pragma once

#include "path.hpp"
#include "types.hpp"

#include <limits>


namespace mpc
{

struct ProjectionResult
{
    size_t segment_index = 0;
    double t = 0.0;  // [0,1] parametric position along segment
    Eigen::Vector2d proj = Eigen::Vector2d::Zero();
};

// Projects the robot onto the path.
// Stateful, forward-only projection with hysteresis and bounded look-ahead.
class Projector
{
public:
    // Reset to the beginning of the path (call when the path changes
    // drastically or the robot is re-localised far from the current position).
    void reset() { last_segment_ = 0; }

    ProjectionResult project(const State& x, const Path& path);

private:
    size_t last_segment_ = 0;
};

}  // namespace mpc
