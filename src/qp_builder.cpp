// QPBuilder::build - assemble the MPC QP.
// Decision vector: z = [ x_0...x_N, u_0...u_{N-1}, eps_0...eps_N ].
// Constraints: dynamics, initial condition, control bounds, accel rate,
// corridor (soft) constraints, slack non-negativity, terminal velocity.
// Sparsity invariant: A's nonzero pattern is constant across iterations.

#include "mpc/qp_builder.hpp"

#include <cmath>
#include <cassert>


namespace mpc
{

static constexpr double kInf = 1e30;

QP QPBuilder::build(
    const State& x0,
    const Control& u_prev,
    const Reference& ref,
    const LinModel& model,
    const QPContext& ctx) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    assert((int)model.A.size() == N && "LinModel horizon mismatch");
    assert((int)ref.x_ref.size() == N + 1 && "x_ref size mismatch");
    assert((int)ref.seg_normals.size() == N + 1 && "seg_normals size mismatch");
    assert((int)ref.proj_pts.size() == N + 1 && "proj_pts size mismatch");

    // Variable counts
    // Bug #4 fix: slack variables cover k = 0..N (N+1 total) so that the
    // terminal state x_N is corridor-constrained via its own slack eps_N.
    const int n_vars = (N + 1) * NX + N * NU + (N + 1);

    // Constraint row layout
    const int row_dyn = 0;
    const int row_ic = row_dyn + N * NX;
    const int row_ubnd = row_ic + NX;
    const int row_accel = row_ubnd + N * NU;
    const int row_corr = row_accel + N * NU;       // 2*(N+1) rows
    const int row_slack = row_corr + 2 * (N + 1);  // (N+1)   rows
    const int row_term = row_slack + (N + 1);
    const int n_constr = row_term + 1;

    // Allocate
    QP qp;
    qp.P.resize(n_vars, n_vars);
    qp.q = Eigen::VectorXd::Zero(n_vars);
    qp.A.resize(n_constr, n_vars);
    qp.l = Eigen::VectorXd::Constant(n_constr, -kInf);
    qp.u = Eigen::VectorXd::Constant(n_constr, kInf);

    std::vector<Eigen::Triplet<double>> Pt, At;
    Pt.reserve(n_vars * 8);
    At.reserve(n_constr * 12);

    // COST (OSQP minimises 0.5 z^T P z + q^T z)
    // For W*(z_i - r_i)^2 : P(i,i) += 2*W, q(i) -= 2*W*r_i

    // 1a. Tracking
    for (int k = 0; k <= N; ++k)
    {
        const int ix = idx_x(k);
        const bool term = (k == N);

        // Weights: terminal weights are elevated, intermediate steps use
        // the configured Q_xy / Q_theta values (no per-step decay).
        const double Qxy = term ? params_.Q_xy_terminal : params_.Q_xy;
        const double Qth = term ? ctx.Q_theta_terminal_eff : ctx.Q_theta_eff;

        Pt.emplace_back(ix + 0, ix + 0, 2.0 * Qxy);
        Pt.emplace_back(ix + 1, ix + 1, 2.0 * Qxy);
        Pt.emplace_back(ix + 2, ix + 2, 2.0 * Qth);

        qp.q(ix + 0) -= 2.0 * Qxy * ref.x_ref[k].x;
        qp.q(ix + 1) -= 2.0 * Qxy * ref.x_ref[k].y;
        qp.q(ix + 2) -= 2.0 * Qth * ref.x_ref[k].theta;
    }

    // 1b. Control effort + smoothness
    //
    // Per-step cost: (R + R_rate) * u_k^2 - 2 * R_rate * u_k * u_{k-1}
    //
    // k = 0:   u_{-1} = u_prev (constant).  Cross-term enters q only.
    // k > 0:   u_{k-1} is a variable.       Off-diagonal P entry (upper-tri).
    for (int k = 0; k < N; ++k)
    {
        const int iu = idx_u(k, N);

        Pt.emplace_back(iu + 0, iu + 0, 2.0 * (params_.R_v + params_.R_rate_v));
        Pt.emplace_back(
            iu + 1,
            iu + 1,
            2.0 * (params_.R_omega + params_.R_rate_omega));

        if (k == 0)
        {
            qp.q(iu + 0) -= 2.0 * params_.R_rate_v * u_prev.v;
            qp.q(iu + 1) -= 2.0 * params_.R_rate_omega * u_prev.omega;
        }
        else
        {
            const int ip = idx_u(k - 1, N);  // ip < iu  - upper-triangular
            Pt.emplace_back(ip + 0, iu + 0, -2.0 * params_.R_rate_v);
            Pt.emplace_back(ip + 1, iu + 1, -2.0 * params_.R_rate_omega);
            Pt.emplace_back(ip + 0, ip + 0, 2.0 * params_.R_rate_v);
            Pt.emplace_back(ip + 1, ip + 1, 2.0 * params_.R_rate_omega);
        }
    }

    // 1c. Slack penalty
    for (int k = 0; k <= N; ++k)  // k=0..N - includes terminal slack eps_N
    {
        Pt.emplace_back(
            idx_slack(k, N),
            idx_slack(k, N),
            2.0 * params_.w_slack);
    }

    qp.P.setFromTriplets(Pt.begin(), Pt.end());

    // =================================================================
    // CONSTRAINTS
    // =================================================================

    // Group 1: Dynamics  x_{k+1} - A_k x_k - B_k u_k = 0
    //
    // All nine A_k entries and all six B_k entries are always emitted,
    // even when some are zero.  This is the critical requirement for a
    // stable sparsity pattern across iterations.
    for (int k = 0; k < N; ++k)
    {
        const int row = row_dyn + k * NX;
        const int xk = idx_x(k);
        const int xk1 = idx_x(k + 1);
        const int uk = idx_u(k, N);

        const auto& Ak = model.A[k];
        const auto& Bk = model.B[k];

        for (int i = 0; i < NX; ++i)
        {
            At.emplace_back(row + i, xk1 + i, 1.0);  // +x_{k+1}
            for (int j = 0; j < NX; ++j)
            {
                At.emplace_back(row + i, xk + j, -Ak(i, j));  // -A_k x_k
            }
            for (int j = 0; j < NU; ++j)
            {
                At.emplace_back(row + i, uk + j, -Bk(i, j));  // -B_k u_k
            }
        }
        for (int i = 0; i < NX; ++i)
        {
            qp.l(row + i) = model.d[k](i);
            qp.u(row + i) = model.d[k](i);
        }
    }

    // Group 2: Initial condition  x_0 = x0
    for (int i = 0; i < NX; ++i)
    {
        At.emplace_back(row_ic + i, idx_x(0) + i, 1.0);
    }

    qp.l(row_ic + 0) = qp.u(row_ic + 0) = x0.x;
    qp.l(row_ic + 1) = qp.u(row_ic + 1) = x0.y;
    qp.l(row_ic + 2) = qp.u(row_ic + 2) = x0.theta;

    // Group 3: Control box bounds
    for (int k = 0; k < N; ++k)
    {
        const int row = row_ubnd + k * NU;
        const int uk = idx_u(k, N);

        At.emplace_back(row + 0, uk + 0, 1.0);
        qp.l(row + 0) = params_.v_min;
        qp.u(row + 0) = params_.v_max;

        At.emplace_back(row + 1, uk + 1, 1.0);
        qp.l(row + 1) = -params_.omega_max;
        qp.u(row + 1) = params_.omega_max;
    }

    // Group 4: Acceleration rate
    const double dv = params_.a_max * dt;
    const double dom = params_.alpha_max * dt;

    for (int k = 0; k < N; ++k)
    {
        const int row = row_accel + k * NU;
        const int uk = idx_u(k, N);

        // v: single-variable row for k=0, two-variable for k>0
        At.emplace_back(row + 0, uk + 0, 1.0);
        if (k == 0)
        {
            qp.l(row + 0) = u_prev.v - dv;
            qp.u(row + 0) = u_prev.v + dv;
        }
        else
        {
            At.emplace_back(row + 0, idx_u(k - 1, N) + 0, -1.0);
            qp.l(row + 0) = -dv;
            qp.u(row + 0) = dv;
        }

        // omega
        At.emplace_back(row + 1, uk + 1, 1.0);
        if (k == 0)
        {
            qp.l(row + 1) = u_prev.omega - dom;
            qp.u(row + 1) = u_prev.omega + dom;
        }
        else
        {
            At.emplace_back(row + 1, idx_u(k - 1, N) + 1, -1.0);
            qp.l(row + 1) = -dom;
            qp.u(row + 1) = dom;
        }
    }

    // Group 5: Soft corridor
    //
    //  e_k = n_k^T * [x_k, y_k] - c_k     (c_k = n_k^T * proj_k)
    //  |e_k| <= d_k + eps_k
    //
    //  Row A:  +n^T x - eps <= d_k + c
    //  Row B:  -n^T x - eps <= d_k - c
    //
    // Funnel fix: d_k is the per-step effective width computed from the
    //   initial exceedance.  Previously ctx.d_hard_eff was written to qp.u()
    //   instead of d_k, making the funneling calculation dead code.

    const double initial_exceedance =
        std::max(0.0, std::abs(ctx.cte_raw) - ctx.d_hard_eff);

    for (int k = 0; k <= N; ++k)
    {
        const int ix = idx_x(k);
        const int is = idx_slack(k, N);
        const int rA = row_corr + 2 * k;
        const int rB = rA + 1;
        const Eigen::Vector2d& n = ref.seg_normals[k];
        const double c = n.dot(ref.proj_pts[k]);

        // Exponentially tighten the corridor from the initial exceedance back
        // down to d_hard_eff over funnel_decay_tau steps.
        const double d_k =
            ctx.d_hard_eff +
            initial_exceedance * std::exp(-k / params_.funnel_decay_tau);

        At.emplace_back(rA, ix + 0, n.x());
        At.emplace_back(rA, ix + 1, n.y());
        At.emplace_back(rA, is, -1.0);
        qp.u(rA) = d_k + c;

        At.emplace_back(rB, ix + 0, -n.x());
        At.emplace_back(rB, ix + 1, -n.y());
        At.emplace_back(rB, is, -1.0);
        qp.u(rB) = d_k - c;
    }

    // Group 6: Slack non-negativity
    for (int k = 0; k <= N; ++k)  // k=0..N - mirrors the corridor loop
    {
        At.emplace_back(row_slack + k, idx_slack(k, N), 1.0);
        qp.l(row_slack + k) = 0.0;
    }

    // Group 7: Terminal velocity
    //
    // Row is always present to keep sparsity constant.
    //   near_goal = false -> v_{N-1} <= v_max (inactive)
    //   near_goal = true  -> v_{N-1} = 0 (enforce stop)
    At.emplace_back(row_term, idx_u(N - 1, N) + 0, 1.0);
    qp.l(row_term) = params_.v_min;
    qp.u(row_term) = ctx.near_goal ? 0.0 : params_.v_max;

    qp.A.setFromTriplets(At.begin(), At.end());
    return qp;
}

}  // namespace mpc
