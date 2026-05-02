#pragma once

#include <cmath>

namespace mpc
{

struct MPCParams
{
    // ── Horizon ───────────────────────────────────────────────────────
    int N = 15;
    double dt = 0.05;  // timestep [s] → 20 Hz

    // ── Velocity limits ──────────────────────────────────────────────
    double v_max = 1.2;      // max forward speed   [m/s]
    double v_min = -0.1;     // min forward speed   [m/s]
    double omega_max = 1.5;  // max angular speed   [rad/s]

    // ── Acceleration limits ──────────────────────────────────────────
    double a_max = 2.0;      // linear accel bound  [m/s²]
    double alpha_max = 2.0;  // angular accel bound [rad/s²]

    // ── Corridor ─────────────────────────────────────────────────────
    double d_hard = 0.05;     // corridor half-width [m]
    double w_slack = 5000.0;  // quadratic penalty on corridor slack ε_k

    // ── Initial-exceedance funnel ─────────────────────────────────────
    // When the robot starts outside the corridor the bound is widened at
    // k=0 to exactly swallow the error and decays back to d_hard over
    // funnel_decay_tau steps so that feasibility is maintained without
    // permanently relaxing the constraint.
    double funnel_decay_tau = 10.0;

    // ── Frenet tracking cost ──────────────────────────────────────────
    // Cross-track error  e_y  (lateral deviation from path centreline).
    double Q_ey = 200.0;           // intermediate step weight
    double Q_ey_terminal = 400.0;  // terminal step weight

    // Heading error  e_theta  (robot heading minus path tangent heading).
    double Q_eth = 10.0;           // intermediate step weight
    double Q_eth_terminal = 50.0;  // terminal step weight

    // Velocity tracking  (v_k − v_ref_k)²
    double Q_v = 5.0;

    // ── Progress reward ───────────────────────────────────────────────
    // Linear term  −w_progress · s_N  in the cost.  Rewards the optimizer
    // for advancing along the path rather than stalling.
    double w_progress = 0.2;

    // ── Control cost ──────────────────────────────────────────────────
    double R_v = 0.5;      // effort on v
    double R_omega = 0.5;  // effort on ω

    // ── Smoothness cost ───────────────────────────────────────────────
    double R_rate_v = 2.0;      // weight on (v_k − v_{k-1})²
    double R_rate_omega = 2.0;  // weight on (ω_k − ω_{k-1})²

    // ── Near-goal detection ───────────────────────────────────────────
    double goal_threshold =
        0.03;                     // remaining arc below which v_N=0 is enforced
    double goal_cte_scale = 2.0;  // |e_y| must be < goal_cte_scale·d_hard

    // ── Solver failure fallback ───────────────────────────────────────
    double fallback_decay = 0.8;

    // ── Derived ───────────────────────────────────────────────────────
    int minBrakingSteps() const
    {
        const double dv = v_max - std::max(v_min, 0.0);
        return static_cast<int>(std::ceil(dv / (a_max * dt)));
    }
};

}  // namespace mpc
