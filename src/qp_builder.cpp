#include "mpc/qp_builder.hpp"


inline int idx_x(int k) { return 3 * k; }
inline int idx_u(int k, int N) { return 3 * (N + 1) + 2 * k; }

namespace mpc
{

QP QPBuilder::build(
    const State& x0,
    const Reference& ref,
    const LinModel& model)
{
    int nx = 3;  // state dimension (x, y, theta)
    int nu = 2;  // control dimension (v, omega)

    int n_state = (N + 1) * nx;
    int n_ctrl = N * nu;
    int n = n_state + n_ctrl;

    int n_ctrl_constr = N * nu;  // 2 per step
    int n_constr = N * nx + nx + n_ctrl_constr;
    int base = N * nx;  // start of initial condition rows

    int base_dyn = 0;
    int base_init = N * nx;
    int base_ctrl = base_init + nx;

    QP qp;
    qp.P.resize(n, n);
    qp.q = Eigen::VectorXd::Zero(n);

    std::vector<Eigen::Triplet<double>> P_triplets;

    // simple tracking cost
    for (int k = 0; k <= N; ++k)
    {
        int ix = idx_x(k);

        P_triplets.emplace_back(ix + 0, ix + 0, 10.0);
        P_triplets.emplace_back(ix + 1, ix + 1, 10.0);
        P_triplets.emplace_back(ix + 2, ix + 2, 1.0);

        const auto& r = ref.x_ref[std::min(k, (int)ref.x_ref.size() - 1)];

        qp.q(ix + 0) = -10.0 * r.x;
        qp.q(ix + 1) = -10.0 * r.y;
        qp.q(ix + 2) = -1.0 * r.theta;
    }

    for (int k = 0; k < N; ++k)
    {
        int iu = idx_u(k, N);

        P_triplets.emplace_back(iu + 0, iu + 0, 0.1);  // v
        P_triplets.emplace_back(iu + 1, iu + 1, 0.1);  // ω
    }

    qp.P.setFromTriplets(P_triplets.begin(), P_triplets.end());

    qp.A.resize(n_constr, n);
    qp.l = Eigen::VectorXd::Zero(n_constr);
    qp.u = Eigen::VectorXd::Zero(n_constr);

    std::vector<Eigen::Triplet<double>> A_triplets;

    for (int k = 0; k < N; ++k)
    {
        int row = k * nx;

        int xk = idx_x(k);
        int xk1 = idx_x(k + 1);
        int uk = idx_u(k, N);

        const auto& A = model.A[k];  // 3x3
        const auto& B = model.B[k];  // 3x2

        for (int i = 0; i < nx; ++i)
        {
            // x_{k+1}
            A_triplets.emplace_back(row + i, xk1 + i, 1.0);

            // -A_k x_k
            for (int j = 0; j < nx; ++j)
            {
                if (A(i, j) != 0.0)
                {
                    A_triplets.emplace_back(row + i, xk + j, -A(i, j));
                }
            }

            // -B_k u_k
            for (int j = 0; j < nu; ++j)
            {
                if (B(i, j) != 0.0)
                {
                    A_triplets.emplace_back(row + i, uk + j, -B(i, j));
                }
            }
        }
    }

    for (int i = 0; i < nx; ++i)
    {
        A_triplets.emplace_back(base + i, idx_x(0) + i, 1.0);
    }

    double v_max = 1.0;  // tune
    double w_max = 1.0;  // tune

    for (int k = 0; k < N; ++k)
    {
        int row = base_ctrl + k * nu;
        int uk = idx_u(k, N);

        // v_k bounds
        A_triplets.emplace_back(row + 0, uk + 0, 1.0);
        qp.l(row + 0) = 0.0;
        qp.u(row + 0) = v_max;

        // ω_k bounds
        A_triplets.emplace_back(row + 1, uk + 1, 1.0);
        qp.l(row + 1) = -w_max;
        qp.u(row + 1) = w_max;
    }

    // assuming State has x, y, theta
    qp.l(base + 0) = x0.x;
    qp.l(base + 1) = x0.y;
    qp.l(base + 2) = x0.theta;

    qp.u(base + 0) = x0.x;
    qp.u(base + 1) = x0.y;
    qp.u(base + 2) = x0.theta;

    qp.A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    return qp;
}

}  // namespace mpc
