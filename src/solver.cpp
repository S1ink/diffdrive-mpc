#include "mpc/solver.hpp"
#include <vector>
#include <iostream>

namespace mpc
{

// Convert Eigen sparse → OSQP CSC
static OSQPCscMatrix* eigenToCSC(const Eigen::SparseMatrix<double>& mat)
{
    Eigen::SparseMatrix<double> A = mat;
    A.makeCompressed();

    OSQPInt nnz = (OSQPInt)A.nonZeros();
    OSQPInt n = (OSQPInt)A.cols();
    OSQPInt m = (OSQPInt)A.rows();

    // Allocate new arrays
    OSQPFloat* values = (OSQPFloat*)malloc(sizeof(OSQPFloat) * nnz);
    OSQPInt* rowind = (OSQPInt*)malloc(sizeof(OSQPInt) * nnz);
    OSQPInt* colptr = (OSQPInt*)malloc(sizeof(OSQPInt) * (n + 1));

    // Copy values
    for (OSQPInt i = 0; i < nnz; ++i)
    {
        values[i] = (OSQPFloat)A.valuePtr()[i];
    }

    // Convert indices (int → OSQPInt)
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

Solver::Solver() : solver_(nullptr), n_(0), m_(0) {}

Solver::~Solver()
{
    if (solver_)
    {
        osqp_cleanup(solver_);
    }
}

void Solver::setup(const QP& qp, int N)
{
    n_ = qp.P.rows();
    m_ = qp.A.rows();
    N_ = N;

    OSQPCscMatrix* P = eigenToCSC(qp.P);
    OSQPCscMatrix* A = eigenToCSC(qp.A);

    std::vector<OSQPFloat> q(qp.q.size());
    std::vector<OSQPFloat> l(qp.l.size());
    std::vector<OSQPFloat> u(qp.u.size());

    for (int i = 0; i < qp.q.size(); ++i)
    {
        q[i] = qp.q[i];
    }
    for (int i = 0; i < qp.l.size(); ++i)
    {
        l[i] = qp.l[i];
    }
    for (int i = 0; i < qp.u.size(); ++i)
    {
        u[i] = qp.u[i];
    }

    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);

    settings->verbose = false;
    settings->warm_starting = 1;

    std::cout << "P nnz: " << qp.P.nonZeros() << std::endl;
    std::cout << "A nnz: " << qp.A.nonZeros() << std::endl;

    osqp_setup(&solver_, P, q.data(), A, l.data(), u.data(), m_, n_, settings);
}

Control Solver::solve()
{
    osqp_solve(solver_);

    Control u{0.0, 0.0};

    // if (solver_->info->status_val != OSQP_SOLVED)
    // {
    //     std::cerr << "OSQP failed: " << solver_->info->status << std::endl;
    //     return u;
    // }

    // for (int i = 0; i < 10; ++i) {
    //     std::cout << solver_->solution->x[i] << std::endl;
    // }

    if (!solver_ || !solver_->solution || !solver_->solution->x)
    {
        return u;
    }

    auto* x = solver_->solution->x;

    int offset = 3 * (N_ + 1);

    u.v = x[offset + 0];
    u.omega = x[offset + 1];

    return u;
}

}  // namespace mpc
