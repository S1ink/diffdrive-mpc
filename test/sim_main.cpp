#include <iostream>
#include <vector>
#include <cmath>

#include "mpc/qp_builder.hpp"
#include "mpc/solver.hpp"
#include "mpc/types.hpp"

using namespace mpc;

static constexpr double DT = 0.05;
static constexpr int N = 10;

// simple forward simulation (nonlinear)
State step(const State& x, const Control& u)
{
    State xn;

    xn.x = x.x + u.v * std::cos(x.theta) * DT;
    xn.y = x.y + u.v * std::sin(x.theta) * DT;
    xn.theta = x.theta + u.omega * DT;

    return xn;
}

// simple straight-line reference
Reference buildReference(const State& x)
{
    Reference ref;
    ref.x_ref.resize(N + 1);

    for (int k = 0; k <= N; ++k)
    {
        ref.x_ref[k].x = x.x + 0.1 * k;
        ref.x_ref[k].y = 0.0;
        ref.x_ref[k].theta = 0.0;
    }

    return ref;
}

// trivial linearization (identity-ish for now)
LinModel buildModel(const State& x)
{
    LinModel model;
    model.A.resize(N);
    model.B.resize(N);

    for (int k = 0; k < N; ++k)
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 2> B = Eigen::Matrix<double, 3, 2>::Zero();

        double theta = x.theta;

        B(0, 0) = std::cos(theta) * DT;
        B(1, 0) = std::sin(theta) * DT;
        B(2, 1) = DT;

        model.A[k] = A;
        model.B[k] = B;
    }

    return model;
}

int main()
{
    QPBuilder builder;
    Solver solver;

    State x;
    x.x = 0.0;
    x.y = 0.0;
    x.theta = 0.0;

    for (int t = 0; t < 200; ++t)
    {
        // build inputs
        Reference ref = buildReference(x);
        LinModel model = buildModel(x);

        // build QP
        QP qp = builder.build(x, ref, model);

        // solve
        solver.setup(qp, N);
        Control u = solver.solve();

        // simulate
        x = step(x, u);

        // print
        std::cout << u.v << ", " << u.omega << std::endl;
        std::cout << "t=" << t << " | x=" << x.x << " y=" << x.y
                  << " th=" << x.theta << " | v=" << u.v << " w=" << u.omega
                  << std::endl;
    }

    return 0;
}
