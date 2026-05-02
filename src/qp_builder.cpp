// =============================================================================
// QPBuilder::build  —  Frenet-frame MPC QP assembly
//
// Decision vector layout:
//   z = [ s_0, e_y,0, e_θ,0  …  s_N, e_y,N, e_θ,N   (NX=3 per step, N+1)
//         v_0, ω_0            …  v_{N-1}, ω_{N-1}    (NU=2 per step, N)
//         ε_0                 …  ε_N                  (1    per step, N+1) ]
//
// Cost:
//   Σ_k  Q_ey·e_y,k²  +  Q_eth·e_θ,k²          (track path centreline)
//   Σ_k  Q_v·(v_k − v_ref,k)²                   (track velocity profile)
//   Σ_k  R_v·v_k² + R_ω·ω_k²                    (control effort)
//   Σ_k  R_rate·(u_k − u_{k-1})²                (smoothness)
//   Σ_k  w_slack·ε_k²                            (soft-corridor penalty)
//   −w_progress · s_N                            (progress reward)
//   Terminal weights Q_ey_terminal, Q_eth_terminal at k=N.
//
// Corridor:
//   |e_y,k| ≤ d_k + ε_k   (box constraint on a state variable — exact,
//                            no normal-vector approximation)
//   d_k = d_hard + max(0, |e_y,0|−d_hard)·exp(−k/funnel_decay_tau)
//
// SPARSITY INVARIANT:
//   All entries of A_k and B_k in Group 1 are emitted unconditionally.
//   Every other group emits a fixed number of entries per row.
//   The nonzero (row,col) pattern of A is therefore identical across all
//   iterations, allowing Solver to use osqp_update_data_mat() instead of
//   a full factorisation every cycle.
// =============================================================================

#include "mpc/qp_builder.hpp"

#include <cassert>
#include <cmath>

namespace mpc
{

static constexpr double kInf = 1e30;

QP QPBuilder::build(
    const FrenetState& x0,
    const Control& u_prev,
    const Reference& ref,
    const LinModel& model,
    const QPContext& ctx) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    assert((int)model.A.size() == N);
    assert((int)ref.v_profile.size() == N + 1);

    // ── Variable counts ───────────────────────────────────────────────
    const int n_vars = (N + 1) * NX + N * NU + (N + 1);

    // ── Constraint row layout ─────────────────────────────────────────
    const int row_dyn = 0;
    const int row_ic = row_dyn + N * NX;
    const int row_ubnd = row_ic + NX;
    const int row_accel = row_ubnd + N * NU;
    const int row_corr = row_accel + N * NU;      // 2*(N+1) rows
    const int row_slack = row_corr + 2 * (N + 1); // N+1 rows
    const int row_term = row_slack + (N + 1);
    const int row_mono = row_term + 1;             // N rows
    const int n_constr = row_mono + N;

    // ── Allocate ──────────────────────────────────────────────────────
    QP qp;
    qp.P.resize(n_vars, n_vars);
    qp.q = Eigen::VectorXd::Zero(n_vars);
    qp.A.resize(n_constr, n_vars);
    qp.l = Eigen::VectorXd::Constant(n_constr, -kInf);
    qp.u = Eigen::VectorXd::Constant(n_constr, kInf);

    std::vector<Eigen::Triplet<double>> Pt, At;
    Pt.reserve(n_vars * 6);
    At.reserve(n_constr * 8);

    // =================================================================
    // COST  (OSQP minimises  0.5 z^T P z + q^T z)
    //
    // For  W·(z_i − r_i)² :  P(i,i) += 2W,  q(i) -= 2Wr_i
    // =================================================================

    // ── 1a. Frenet tracking ───────────────────────────────────────────
    for (int k = 0; k <= N; ++k)
    {
        const int ix = idx_x(k);
        const bool term = (k == N);

        const double Qey = term ? params_.Q_ey_terminal : params_.Q_ey;
        const double Qeth = term ? params_.Q_eth_terminal : params_.Q_eth;

        // e_y cost   (ix+1 = e_y index)
        Pt.emplace_back(ix + 1, ix + 1, 2.0 * Qey);

        // e_theta cost  (ix+2 = e_theta index)
        Pt.emplace_back(ix + 2, ix + 2, 2.0 * Qeth);
    }

    // ── 1b. Progress reward  −w_progress · s_N ────────────────────────
    qp.q(idx_x(N) + 0) -= params_.w_progress;

    // ── 1c. Control effort + velocity tracking + smoothness ───────────
    //
    // Per-step velocity cost:  (R_v + Q_v)·v_k² − 2·Q_v·v_ref·v_k
    // Per-step omega cost:      R_omega·ω_k²
    // Smoothness:              R_rate·(u_k − u_{k-1})² via off-diagonal P.
    for (int k = 0; k < N; ++k)
    {
        const int iu = idx_u(k, N);
        const double v_ref = ref.v_profile[k];

        // Velocity: effort + profile tracking
        Pt.emplace_back(
            iu + 0,
            iu + 0,
            2.0 * (params_.R_v + params_.Q_v + params_.R_rate_v));
        qp.q(iu + 0) -= 2.0 * params_.Q_v * v_ref;

        // Omega: effort + smoothness
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
            const int ip = idx_u(k - 1, N);  // ip < iu → upper-triangular ✓
            Pt.emplace_back(ip + 0, iu + 0, -2.0 * params_.R_rate_v);
            Pt.emplace_back(ip + 1, iu + 1, -2.0 * params_.R_rate_omega);
            Pt.emplace_back(ip + 0, ip + 0, 2.0 * params_.R_rate_v);
            Pt.emplace_back(ip + 1, ip + 1, 2.0 * params_.R_rate_omega);
        }
    }

    // ── 1d. Slack penalty ─────────────────────────────────────────────
    for (int k = 0; k <= N; ++k)
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

    // ── Group 1: Dynamics  x_{k+1} − A_k x_k − B_k u_k = d_k ───────
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
            At.emplace_back(row + i, xk1 + i, 1.0);
            for (int j = 0; j < NX; ++j)
            {
                At.emplace_back(row + i, xk + j, -Ak(i, j));
            }
            for (int j = 0; j < NU; ++j)
            {
                At.emplace_back(row + i, uk + j, -Bk(i, j));
            }
        }
        for (int i = 0; i < NX; ++i)
        {
            qp.l(row + i) = model.d[k](i);
            qp.u(row + i) = model.d[k](i);
        }
    }

    // ── Group 2: Initial condition  x_0 = x0_frenet ─────────────────
    for (int i = 0; i < NX; ++i)
    {
        At.emplace_back(row_ic + i, idx_x(0) + i, 1.0);
    }
    qp.l(row_ic + 0) = qp.u(row_ic + 0) = x0.s;
    qp.l(row_ic + 1) = qp.u(row_ic + 1) = x0.e_y;
    qp.l(row_ic + 2) = qp.u(row_ic + 2) = x0.e_theta;

    // ── Group 3: Control box bounds ───────────────────────────────────
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

    // ── Group 4: Acceleration rate ────────────────────────────────────
    const double dv = params_.a_max * dt;
    const double dom = params_.alpha_max * dt;

    for (int k = 0; k < N; ++k)
    {
        const int row = row_accel + k * NU;
        const int uk = idx_u(k, N);

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

    // ── Group 5: Soft corridor  |e_y,k| ≤ d_k + ε_k ─────────────────
    //
    // e_y is at decision-variable index idx_x(k)+1.
    // The box constraint becomes two rows:
    //   Row A:  +e_y,k − ε_k ≤  d_k
    //   Row B:  −e_y,k − ε_k ≤  d_k
    //
    // d_k funnels from |e_y,0| at k=0 back to d_hard over funnel_decay_tau.
    // This maintains feasibility when the robot starts outside the corridor
    // without permanently relaxing the constraint.
    const double initial_exceedance =
        std::max(0.0, std::abs(ctx.e_y_0) - params_.d_hard);

    for (int k = 0; k <= N; ++k)
    {
        const int ix = idx_x(k);
        const int is = idx_slack(k, N);
        const int rA = row_corr + 2 * k;
        const int rB = rA + 1;

        const double d_k =
            params_.d_hard +
            initial_exceedance * std::exp(-k / params_.funnel_decay_tau);

        At.emplace_back(rA, ix + 1, 1.0);   // +e_y,k
        At.emplace_back(rA, is, -1.0);       // −ε_k
        qp.u(rA) = d_k;

        At.emplace_back(rB, ix + 1, -1.0);  // −e_y,k
        At.emplace_back(rB, is, -1.0);       // −ε_k
        qp.u(rB) = d_k;
    }

    // ── Group 6: Slack non-negativity  ε_k ≥ 0 ───────────────────────
    for (int k = 0; k <= N; ++k)
    {
        At.emplace_back(row_slack + k, idx_slack(k, N), 1.0);
        qp.l(row_slack + k) = 0.0;
    }

    // ── Group 7: Terminal velocity ────────────────────────────────────
    At.emplace_back(row_term, idx_u(N - 1, N) + 0, 1.0);
    qp.l(row_term) = params_.v_min;
    qp.u(row_term) = ctx.near_goal ? 0.0 : params_.v_max;

    // ── Group 8: Monotonicity  s_{k+1} − s_k ≥ 0 ─────────────────────
    //
    // Prevents the optimizer from choosing trajectories that back up along
    // the path.  Encoded as two nonzeros per row for fixed sparsity.
    for (int k = 0; k < N; ++k)
    {
        const int row = row_mono + k;
        At.emplace_back(row, idx_x(k + 1) + 0, 1.0);   // +s_{k+1}
        At.emplace_back(row, idx_x(k) + 0, -1.0);       // −s_k
        qp.l(row) = 0.0;
    }

    qp.A.setFromTriplets(At.begin(), At.end());
    return qp;
}

}  // namespace mpc
