#pragma once

#include "types.hpp"
#include <vector>
#include <Eigen/Dense>

namespace mpc
{

struct LinModel
{
    std::vector<Eigen::Matrix3d> A;
    std::vector<Eigen::Matrix<double, 3, 2>> B;
    std::vector<Eigen::Vector3d> d;  // affine residual
};

/// Linearises the Frenet-frame unicycle dynamics about a warm-start trajectory.
///
/// Frenet dynamics (exact, κ=0 for polyline segments):
///   s_{k+1}       = s_k + v_k·cos(e_θ,k)·dt
///   e_y,{k+1}     = e_y,k + v_k·sin(e_θ,k)·dt
///   e_θ,{k+1}     = e_θ,k + ω_k·dt − Δθ_path,k
///
/// where Δθ_path,k = θ_path(s_{k+1}) − θ_path(s_k) is the path-heading change
/// between consecutive steps (zero within a segment, corner angle at junctions).
///
/// The Jacobians are identical in structure to the global-frame case; the only
/// difference is that e_θ replaces θ and Δθ_path enters the affine residual.
class Linearizer
{
public:
    double dt;

    explicit Linearizer(double dt_) : dt(dt_) {}

    /// @param traj          Warm-start Frenet trajectory, length N.
    /// @param u             Warm-start controls, length N.
    /// @param delta_theta   Path heading change per step, length N.
    ///                      delta_theta[k] = θ_path(s̄_{k+1}) − θ_path(s̄_k).
    ///                      Zero within a segment; equals the corner angle when
    ///                      s̄_k and s̄_{k+1} span a polyline junction.
    LinModel linearize(
        const std::vector<FrenetState>& traj,
        const std::vector<Control>& u,
        const std::vector<double>& delta_theta);
};

}  // namespace mpc
