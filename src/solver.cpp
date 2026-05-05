#include "mpc/solver.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>


namespace mpc
{

// Helpers

// Convert an Eigen sparse matrix (already column-major / CSC) to a heap-
// allocated OSQPCscMatrix.  The caller owns the returned pointer and must
// free it with OSQPCscMatrix_free().
static OSQPCscMatrix* eigenToCSC(const Eigen::SparseMatrix<double>& mat)
{
    Eigen::SparseMatrix<double> A = mat;
    A.makeCompressed();

    const OSQPInt nnz = (OSQPInt)A.nonZeros();
    const OSQPInt n = (OSQPInt)A.cols();
    const OSQPInt m = (OSQPInt)A.rows();

    OSQPFloat* values = (OSQPFloat*)malloc(sizeof(OSQPFloat) * nnz);
    OSQPInt* rowind = (OSQPInt*)malloc(sizeof(OSQPInt) * nnz);
    OSQPInt* colptr = (OSQPInt*)malloc(sizeof(OSQPInt) * (n + 1));

    for (OSQPInt i = 0; i < nnz; ++i)
    {
        values[i] = (OSQPFloat)A.valuePtr()[i];
    }
    for (OSQPInt i = 0; i < nnz; ++i)
    {
        rowind[i] = (OSQPInt)A.innerIndexPtr()[i];
    }
    for (OSQPInt i = 0; i < n + 1; ++i)
    {
        colptr[i] = (OSQPInt)A.outerIndexPtr()[i];
    }

    return OSQPCscMatrix_new(m, n, nnz, values, rowind, colptr);
}

// Constructor / destructor

Solver::Solver() = default;

Solver::~Solver()
{
    if (solver_)
    {
        osqp_cleanup(solver_);
    }
}

// Public API

bool Solver::update(const QP& qp, int N)
{
    const int new_n = qp.P.rows();
    const int new_m = qp.A.rows();

    // Full re-setup if dimensions changed or first call.
    if (!initialized_ || new_n != n_ || new_m != m_ || N != N_)
    {
        fullSetup(qp, N);
    }
    else
    {
        incrementalUpdate(qp);
    }

    // Apply warm start from shifted previous solution
    if (!z_warm_.empty())
    {
        // osqp_warm_start accepts (solver, x, y); y = nullptr -> dual unchanged
        osqp_warm_start(solver_, z_warm_.data(), nullptr);
    }

    // Solve
    osqp_solve(solver_);

    if (solver_->info->status_val != OSQP_SOLVED)
    {
        std::cerr << "[Solver] OSQP status: " << solver_->info->status
                  << " - applying fallback\n";
        return false;
    }

    // Store solution and prepare warm start for next iteration
    const double* x = solver_->solution->x;
    solution_.assign(x, x + n_);

    shiftAndStoreWarmStart();
    return true;
}

Control Solver::getControl() const
{
    if (solution_.empty())
    {
        throw std::runtime_error(
            "Solver::getControl() called before a successful solve");
    }

    // Controls start after (N_+1)*NX states.
    // NX = 3, as defined in qp_builder.hpp
    const int offset = 3 * (N_ + 1);
    return {solution_[offset + 0], solution_[offset + 1]};
}

std::vector<State> Solver::getStatePrediction() const
{
    if (solution_.empty() || N_ == 0)
    {
        return {};
    }

    // State blocks occupy the first (N_+1)*NX elements of the decision vector.
    // Layout: z = [ x_0 x_1 ... x_N  u_0 ... u_{N-1}  eps_0 ... eps_{N-1} ]
    constexpr int NX = 3;
    std::vector<State> pred(N_ + 1);
    for (int k = 0; k <= N_; ++k)
    {
        const int base = NX * k;
        pred[k] = {solution_[base], solution_[base + 1], solution_[base + 2]};
    }
    return pred;
}

// Private helpers

void Solver::fullSetup(const QP& qp, int N)
{
    if (solver_)
    {
        osqp_cleanup(solver_);
        solver_ = nullptr;
        initialized_ = false;
    }

    n_ = qp.P.rows();
    m_ = qp.A.rows();
    N_ = N;

    OSQPCscMatrix* P = eigenToCSC(qp.P);
    OSQPCscMatrix* A = eigenToCSC(qp.A);

    // Fill vector buffers
    q_buf_.resize(qp.q.size());
    l_buf_.resize(qp.l.size());
    u_buf_.resize(qp.u.size());
    for (int i = 0; i < (int)qp.q.size(); ++i)
    {
        q_buf_[i] = (OSQPFloat)qp.q[i];
    }
    for (int i = 0; i < (int)qp.l.size(); ++i)
    {
        l_buf_[i] = (OSQPFloat)qp.l[i];
    }
    for (int i = 0; i < (int)qp.u.size(); ++i)
    {
        u_buf_[i] = (OSQPFloat)qp.u[i];
    }

    // Pre-allocate Ax buffer (same nnz every iteration once pattern is fixed)
    Eigen::SparseMatrix<double> Ac = qp.A;
    Ac.makeCompressed();
    Ax_buf_.resize(Ac.nonZeros());

    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->verbose = false;
    settings->warm_starting = 1;
    settings->eps_abs = 1e-4;
    settings->eps_rel = 1e-4;
    settings->max_iter = 4000;
    settings->polishing = false;  // fast real-time operation

    const int ret = osqp_setup(
        &solver_,
        P,
        q_buf_.data(),
        A,
        l_buf_.data(),
        u_buf_.data(),
        (OSQPInt)m_,
        (OSQPInt)n_,
        settings);

    OSQPCscMatrix_free(P);
    OSQPCscMatrix_free(A);
    free(settings);

    if (ret != 0)
    {
        throw std::runtime_error("osqp_setup() failed");
    }

    z_warm_.clear();  // no warm start for first solve after setup
    solution_.clear();
    initialized_ = true;
}

void Solver::incrementalUpdate(const QP& qp)
{
    // Update linear cost q
    for (int i = 0; i < (int)qp.q.size(); ++i)
    {
        q_buf_[i] = (OSQPFloat)qp.q[i];
    }

    // Update bounds l, u
    for (int i = 0; i < (int)qp.l.size(); ++i)
    {
        l_buf_[i] = (OSQPFloat)qp.l[i];
    }
    for (int i = 0; i < (int)qp.u.size(); ++i)
    {
        u_buf_[i] = (OSQPFloat)qp.u[i];
    }

    // osqp_update_data_vec(solver, q, q_n, l, l_n, u, u_n)
    // Pass nullptr / 0 for any component you don't want to update.
    osqp_update_data_vec(solver_, q_buf_.data(), l_buf_.data(), u_buf_.data());

    // Update A (LTV dynamics + corridor normals change each step)
    // P is constant (weight matrices don't change), so we skip it.
    // IMPORTANT: this assumes the sparsity structure of A is identical to
    // the one used in fullSetup().  QPBuilder guarantees this by always
    // emitting all entries of A_k and B_k regardless of zero values.
    Eigen::SparseMatrix<double> Ac = qp.A;
    Ac.makeCompressed();
    assert(
        (int)Ac.nonZeros() == (int)Ax_buf_.size() &&
        "A sparsity pattern changed - call fullSetup() instead");

    for (int i = 0; i < (int)Ax_buf_.size(); ++i)
    {
        Ax_buf_[i] = (OSQPFloat)Ac.valuePtr()[i];
    }

    // osqp_update_data_mat(solver, Px, Px_idx, P_n, Ax, Ax_idx, A_n)
    // Passing nullptr for index arrays -> update ALL nonzeros in order.
    osqp_update_data_mat(
        solver_,
        nullptr,
        nullptr,
        0,  // P unchanged
        Ax_buf_.data(),
        nullptr,
        (OSQPInt)Ax_buf_.size());
}

void Solver::shiftAndStoreWarmStart()
{
    // Copy solution into the warm-start buffer, then shift forward by one
    // timestep so it is a reasonable initial guess for the next QP.
    //
    // Decision vector layout (mirrored from qp_builder.hpp):
    //   z = [ x_0 ... x_N      (NX=3 per step, N+1 blocks)
    //         u_0 ... u_{N-1}  (NU=2 per step, N   blocks)
    //         eps_0 ... eps_N  (1   per step, N+1 blocks) ]  // Bug #4 fix
    //
    // The slack shift runs N iterations (was N-1): eps_0 <- eps_1, ..., eps_{N-1} <- eps_N.
    // eps_N stays in place (repeat).

    if ((int)solution_.size() != n_)
    {
        return;
    }

    z_warm_.resize(n_);
    for (int i = 0; i < n_; ++i)
    {
        z_warm_[i] = (OSQPFloat)solution_[i];
    }

    // Shift state blocks: x_k <- x_{k+1}, repeat x_N
    for (int k = 0; k < N_; ++k)
    {
        const int dst = mpc::NX * k;
        const int src = mpc::NX * (k + 1);
        for (int i = 0; i < mpc::NX; ++i)
        {
            z_warm_[dst + i] = z_warm_[src + i];
        }
    }
    // x_N already in place (nothing to do)

    // Shift control blocks: u_k <- u_{k+1}, repeat u_{N-1}
    const int ctrl_base = mpc::NX * (N_ + 1);
    for (int k = 0; k < N_ - 1; ++k)
    {
        const int dst = ctrl_base + mpc::NU * k;
        const int src = ctrl_base + mpc::NU * (k + 1);
        for (int i = 0; i < mpc::NU; ++i)
        {
            z_warm_[dst + i] = z_warm_[src + i];
        }
    }
    // u_{N-1} unchanged (repeat)

    // Shift slack blocks: eps_k <- eps_{k+1}, repeat eps_N
    const int slack_base = ctrl_base + mpc::NU * N_;
    for (int k = 0; k < N_; ++k)  // N iterations: covers eps_{N-1} <- eps_N
    {
        z_warm_[slack_base + k] = z_warm_[slack_base + k + 1];
    }
    // eps_N unchanged (repeat)
}

}  // namespace mpc
