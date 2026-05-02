#pragma once

#include "types.hpp"
#include "reference.hpp"
#include "linearization.hpp"
#include "params.hpp"

#include <Eigen/Sparse>

namespace mpc
{

// ── Dimension constants ───────────────────────────────────────────────────────
constexpr int NX = 3;  // Frenet state  [s, e_y, e_θ]
constexpr int NU = 2;  // control       [v, ω]

// ── Decision-vector index helpers ─────────────────────────────────────────────
//
//   z = [ x_0 … x_N       (NX each, N+1 blocks)  ← Frenet states
//       | u_0 … u_{N-1}   (NU each, N   blocks)
//       | ε_0 … ε_N        (1  each, N+1 blocks)  ← corridor slack ]

inline int idx_x(int k) { return NX * k; }
inline int idx_u(int k, int N) { return NX * (N + 1) + NU * k; }
inline int idx_slack(int k, int N) { return NX * (N + 1) + NU * N + k; }

// ── QP data ───────────────────────────────────────────────────────────────────
struct QP
{
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd q;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// ── Per-iteration adaptive context ───────────────────────────────────────────
struct QPContext
{
    /// Initial cross-track error e_y,0 [m].  Used to build the feasibility
    /// funnel: d_k = d_hard + max(0, |e_y_0|−d_hard)·exp(−k/τ).
    double e_y_0 = 0.0;

    /// When true, enforces v_{N-1} = 0 (terminal stop near goal).
    bool near_goal = false;
};

// ── Builder ───────────────────────────────────────────────────────────────────
//
// Decision vector layout:
//   z = [ s_0, e_y,0, e_θ,0,  …  s_N, e_y,N, e_θ,N   (NX=3 per step, N+1)
//         v_0, ω_0,            …  v_{N-1}, ω_{N-1}    (NU=2 per step, N)
//         ε_0,                 …  ε_N                  (1    per step, N+1) ]
//
// Constraint groups (rows of A):
//   1. Dynamics      x_{k+1} = A_k x_k + B_k u_k + d_k     N·NX rows
//   2. Initial cond. x_0 = x0_frenet                        NX rows
//   3. Control box   v_min ≤ v_k ≤ v_max, |ω_k| ≤ ω_max   N·NU rows
//   4. Accel. rate   |u_k − u_{k-1}| ≤ Δu_max              N·NU rows
//   5. Corridor      |e_y,k| ≤ d_k + ε_k                   2·(N+1) rows
//   6. Slack bound   ε_k ≥ 0                                N+1  rows
//   7. Terminal vel  v_{N-1} = 0 or ≤ v_max                 1    row
//   8. Monotonicity  s_{k+1} − s_k ≥ 0                      N    rows
class QPBuilder
{
public:
    explicit QPBuilder(const MPCParams& p) : params_(p) {}

    QP build(
        const FrenetState& x0,
        const Control& u_prev,
        const Reference& ref,
        const LinModel& model,
        const QPContext& ctx = QPContext{}) const;

private:
    MPCParams params_;
};

}  // namespace mpc
