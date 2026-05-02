#include "mpc/linearization.hpp"
#include <cmath>
#include <cassert>

namespace mpc
{

LinModel Linearizer::linearize(
    const std::vector<FrenetState>& traj,
    const std::vector<Control>& u,
    const std::vector<double>& delta_theta)
{
    const int N = (int)traj.size();
    assert((int)u.size() == N);
    assert((int)delta_theta.size() == N);

    LinModel m;
    m.A.resize(N);
    m.B.resize(N);
    m.d.resize(N);

    for (int k = 0; k < N; ++k)
    {
        const double eth = traj[k].e_theta;
        const double v = u[k].v;

        // ── Jacobian A = ∂f/∂x ──────────────────────────────────────
        //
        //   [1   0   −v·sin(ē_θ)·dt ]
        //   [0   1    v·cos(ē_θ)·dt ]
        //   [0   0    1              ]
        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        A(0, 2) = -v * std::sin(eth) * dt;
        A(1, 2) = v * std::cos(eth) * dt;

        // ── Jacobian B = ∂f/∂u ──────────────────────────────────────
        //
        //   [ cos(ē_θ)·dt   0  ]
        //   [ sin(ē_θ)·dt   0  ]
        //   [ 0              dt ]
        Eigen::Matrix<double, 3, 2> B;
        B << std::cos(eth) * dt, 0.0,
             std::sin(eth) * dt, 0.0,
             0.0,                dt;

        // ── Affine residual  d = f(x̄,ū) − A·x̄ − B·ū ───────────────
        //
        //   d_s     =  v·sin(ē_θ)·ē_θ·dt
        //   d_ey    = −v·cos(ē_θ)·ē_θ·dt
        //   d_eth   = −Δθ_path,k
        //
        // The Δθ_path term encodes the path-heading discontinuity at polyline
        // corners: within a segment it is zero; at a junction crossing it equals
        // the signed corner angle.  This is the only place corner geometry enters
        // the linearisation, and it does so exactly without any additional logic.
        m.d[k](0) = v * std::sin(eth) * eth * dt;
        m.d[k](1) = -v * std::cos(eth) * eth * dt;
        m.d[k](2) = -delta_theta[k];

        m.A[k] = A;
        m.B[k] = B;
    }

    return m;
}

}  // namespace mpc
