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

/// Internal state in the Frenet frame of the current path.
///   s       arc-length progress along path [m]
///   e_y     signed cross-track error (left-of-path positive) [m]
///   e_theta heading error = theta_robot − theta_path [rad]
struct FrenetState
{
    double s;
    double e_y;
    double e_theta;
};

struct Control
{
    double v;
    double omega;
};

}  // namespace mpc
