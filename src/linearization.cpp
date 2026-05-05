#include "mpc/linearization.hpp"

#include <cmath>


namespace mpc
{

LinModel Linearizer::linearize(
    const std::vector<State>& traj,
    const std::vector<Control>& u)
{
    LinModel m;
    int N = traj.size();
    m.A.resize(N);
    m.B.resize(N);
    m.d.resize(N);

    for (int k = 0; k < N; ++k)
    {
        double th = traj[k].theta;
        double v = u[k].v;

        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        A(0, 2) = -v * std::sin(th) * dt;
        A(1, 2) = v * std::cos(th) * dt;

        Eigen::Matrix<double, 3, 2> B;
        B << std::cos(th) * dt, 0, std::sin(th) * dt, 0, 0, dt;

        // Affine residual: d = f(x,u) - A*x - B*u (evaluated at linearization point)
        m.d[k] << v * std::sin(th) * th * dt, -v * std::cos(th) * th * dt, 0.0;

        m.A[k] = A;
        m.B[k] = B;
    }
    return m;
}

}  // namespace mpc
