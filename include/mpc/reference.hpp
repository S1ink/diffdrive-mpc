#pragma once

#include "types.hpp"
#include "path.hpp"
#include "projection.hpp"
#include <vector>

namespace mpc
{

struct Reference
{
    std::vector<State> x_ref;
};

class ReferenceGenerator
{
public:
    double dt;
    int N;
    double v_ref;

    Reference generate(const Path& path, const ProjectionResult& proj);
};

}  // namespace mpc
