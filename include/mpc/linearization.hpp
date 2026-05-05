#pragma once

#include "types.hpp"

#include <Eigen/Dense>

#include <vector>


namespace mpc
{

struct LinModel
{
    std::vector<Eigen::Matrix3d> A;
    std::vector<Eigen::Matrix<double, 3, 2>> B;
    std::vector<Eigen::Vector3d> d;  // affine residual
};

class Linearizer
{
public:
    double dt;

    explicit Linearizer(double dt) : dt(dt) {}

    LinModel linearize(
        const std::vector<State>& traj,
        const std::vector<Control>& u);
};

}  // namespace mpc
