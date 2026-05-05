#pragma once

#include <Eigen/Dense>


namespace mpc
{

struct State
{
    double x;
    double y;
    double theta;
};

struct Control
{
    double v;
    double omega;
};

}  // namespace mpc
