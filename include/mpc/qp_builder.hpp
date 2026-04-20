#pragma once

#include "types.hpp"
#include "reference.hpp"
#include "linearization.hpp"
#include <Eigen/Sparse>

namespace mpc
{

struct QP
{
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd q;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l, u;
};

class QPBuilder
{
public:
    int N;

    QP build(const State& x0, const Reference& ref, const LinModel& model);
};

}  // namespace mpc
