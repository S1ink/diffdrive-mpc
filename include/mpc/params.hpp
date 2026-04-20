#pragma once

namespace mpc
{

/// Central parameter struct — pass by value or const-ref throughout.
/// Every tunable number lives here so nothing is buried in .cpp files.
struct MPCParams
{
    // ── Horizon ──────────────────────────────────────────────────────
    int N = 15;        // prediction horizon (steps)
    double dt = 0.05;  // timestep [s]  → 20 Hz

    // ── Velocity limits ───────────────────────────────────────────────
    double v_max = 0.8;      // max forward speed   [m/s]
    double v_min = 0.0;      // min forward speed   [m/s]  (no reverse)
    double omega_max = 1.2;  // max angular speed   [rad/s]

    // ── Acceleration limits ───────────────────────────────────────────
    double a_max = 0.5;      // linear accel bound  [m/s²]
    double alpha_max = 2.0;  // angular accel bound [rad/s²]

    // ── Reference cruise speed ────────────────────────────────────────
    double v_ref = 0.4;  // nominal forward speed [m/s]
    // actual speed per step is min(v_ref, curvature-limit, stop-limit)

    // ── Corridor / soft constraint ────────────────────────────────────
    double d_hard = 0.08;  // hard corridor half-width [m]
    //   = path_tolerance + noise_margin  (e.g. 5 cm + 3 cm)
    double w_slack = 2000.0;  // quadratic penalty on corridor slack ε_k

    // ── Tracking cost ─────────────────────────────────────────────────
    double Q_xy = 20.0;             // position weight (intermediate steps)
    double Q_theta = 2.0;           // heading weight  (intermediate steps)
    double Q_xy_terminal = 80.0;    // elevated position weight at step N
    double Q_theta_terminal = 8.0;  // elevated heading weight  at step N

    // ── Control cost ─────────────────────────────────────────────────
    double R_v = 0.5;      // effort on v
    double R_omega = 0.5;  // effort on ω

    // ── Smoothness cost (penalises Δu between consecutive steps) ──────
    double R_rate_v = 2.0;      // weight on (v_k − v_{k-1})²
    double R_rate_omega = 2.0;  // weight on (ω_k − ω_{k-1})²

    // ── Projection hysteresis ─────────────────────────────────────────
    // Advance to next segment only when parametric t exceeds this threshold.
    double seg_advance_t = 0.7;

    // ── Noise deadband ────────────────────────────────────────────────
    double d_deadband = 0.02;  // ignore tracking errors below this [m]

    // ── Adaptive corridor ─────────────────────────────────────────────
    // When cross-track error exceeds d_hard, scale the corridor width up
    // by this factor so the solver remains feasible while recovering.
    double adaptive_corridor_scale = 1.8;

    // ── Velocity reduction under error ────────────────────────────────
    // v_ref_k *= clamp(1 − v_error_gain * |cte|,  v_min_scale, 1)
    double v_error_gain = 3.0;
    double v_min_scale = 0.2;

    // ── Reference blending ────────────────────────────────────────────
    // Smooths abrupt path changes: ref = α·new + (1−α)·old
    double blend_alpha = 0.7;  // 1.0 = no blending (pure new reference)

    // ── Heading weight scaling ────────────────────────────────────────
    // Q_theta_eff = Q_theta * exp(−heading_scale_k * |cte|)
    // Prevents spinning in place instead of recovering lateral position.
    double heading_scale_k = 8.0;

    // ── Path-change / projection reset ───────────────────────────────
    // If the nearest point on the new path is more than this far from the
    // robot's last projected position, reset the projector's segment state.
    double path_reset_threshold = 0.5;  // [m]

    // ── Near-goal detection ───────────────────────────────────────────
    // When remaining path length drops below this, enforce v_N = 0.
    double goal_threshold = 0.3;  // [m]

    // ── Solver failure fallback ───────────────────────────────────────
    // On OSQP failure, return u_prev scaled by this factor.
    double fallback_decay = 0.8;
};

}  // namespace mpc
