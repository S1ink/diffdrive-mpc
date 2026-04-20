#pragma once

#include "types.hpp"
#include "path.hpp"

namespace mpc
{

struct ProjectionResult
{
    size_t segment_index;
    double t;  // [0,1] along segment
    Eigen::Vector2d proj;
};

class Projector
{
public:
    ProjectionResult project(const State& x, const Path& path);
};

}  // namespace mpc
