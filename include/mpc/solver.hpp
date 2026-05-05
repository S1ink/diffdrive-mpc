#pragma once

#include "types.hpp"
#include "qp_builder.hpp"

#include <osqp.h>

#include <vector>


namespace mpc
{

// OSQP wrapper with:
//   - Persistent setup: re-factorizes only when problem dimensions change.
//     Per-iteration calls use osqp_update_data_mat / osqp_update_data_vec
//     which skip the KKT factorization when P is unchanged.
//   - Warm start: the previous primal solution is shifted by one timestep
//     and used to seed each new solve.
//   - Failure handling: returns false from update(); caller applies fallback.
//
// Typical usage:
//   Solver solver;
//   ...
//   if (!solver.update(qp, N)) {
//       u = u_prev * fallback_decay;   // apply fallback externally
//   } else {
//       u = solver.getControl();
//   }
class Solver
{
public:
    Solver();
    ~Solver();

    // Build / update the internal OSQP problem and solve.
    //
    // On the first call (or when N changes), performs a full osqp_setup().
    // On subsequent calls with the same N, uses incremental update APIs so
    // that only the KKT vector updates (not a full refactorisation) are
    // needed when only q/l/u change.
    //
    // @return true  if OSQP reported OSQP_SOLVED.
    // @return false otherwise (caller should apply a fallback control).
    bool update(const QP& qp, int N);

    // Return the first control (u_0) from the most recent successful solve.
    // Undefined if the last update() returned false.
    Control getControl() const;

    // Return the full predicted state trajectory x_0 ... x_N extracted from
    // the most recent solution vector.  Returns an empty vector if no
    // successful solve has occurred yet.
    std::vector<State> getStatePrediction() const;

private:
    OSQPSolver* solver_ = nullptr;

    bool initialized_ = false;
    int n_ = 0;  // total decision variables
    int m_ = 0;  // total constraints
    int N_ = 0;  // MPC horizon

    // Primal warm-start buffer (shifted solution from previous iteration)
    std::vector<OSQPFloat> z_warm_;

    // Per-iteration update buffers (avoid repeated heap allocation)
    std::vector<OSQPFloat> q_buf_;
    std::vector<OSQPFloat> l_buf_;
    std::vector<OSQPFloat> u_buf_;
    std::vector<OSQPFloat> Ax_buf_;  // nonzero values of A

    // Most recent solution (used by getControl / getStatePrediction)
    std::vector<double> solution_;

    // Private helpers
    void fullSetup(const QP& qp, int N);
    void incrementalUpdate(const QP& qp);
    void shiftAndStoreWarmStart();
};

}  // namespace mpc