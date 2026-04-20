#pragma once
#include "qp_builder.hpp"
#include "types.hpp"

#include <osqp.h>  // <-- include directly

namespace mpc
{

class Solver
{
public:
    Solver();
    ~Solver();

    void setup(const QP& qp, int N);
    Control solve();

private:
    OSQPSolver* solver_;  // <-- new API

    int n_;
    int m_;
    int N_;
};

}  // namespace mpc
