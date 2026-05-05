#pragma once

#include "types.hpp"
#include "params.hpp"
#include "reference.hpp"
#include "linearization.hpp"

#include <Eigen/Sparse>


namespace mpc
{

// Dimension constants
constexpr int NX = 3;  // state  [x, y, theta]
constexpr int NU = 2;  // control [v, omega]

// Decision-vector index helpers
// z = [ x_0 ... x_N (NX each, N+1 blocks)
//       u_0 ... u_{N-1} (NU each, N blocks)
//       eps_0 ... eps_N (1 each, N+1 blocks)  // corridor slack ]

inline int idx_x(int k) { return NX * k; }
inline int idx_u(int k, int N) { return NX * (N + 1) + NU * k; }
inline int idx_slack(int k, int N) { return NX * (N + 1) + NU * N + k; }


// QP data
struct QP
{
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd q;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// Per-iteration adaptive context
struct QPContext
{
    // Effective corridor half-width [m].  Widened during recovery.
    double d_hard_eff = 0.08;

    // Raw initial cross-track error [m] at step k=0. Used to build
    // the dynamic feasibility funnel.
    double cte_raw = 0.0;

    // Effective heading weight.  Reduced when far from path to prevent
    // spinning in place instead of recovering lateral position.
    double Q_theta_eff = 2.0;

    // Effective terminal heading weight (scaled like Q_theta_eff).
    double Q_theta_terminal_eff = 8.0;

    // When true, enforces v_{N-1} = 0 (terminal stop near goal).
    bool near_goal = false;
};


// Builder
class QPBuilder
{
public:
    explicit QPBuilder(const MPCParams& p) : params_(p) {}

    // Assemble the QP for one MPC iteration.
    //
    // Sparsity guarantee: the returned QP always has the SAME nonzero
    // pattern in A regardless of model / ref / ctx values.  This allows
    // Solver to use osqp_update_data_mat() instead of a full re-setup.
    QP build(
        const State& x0,
        const Control& u_prev,
        const Reference& ref,
        const LinModel& model,
        const QPContext& ctx = QPContext{}) const;

private:
    MPCParams params_;
};

}  // namespace mpc