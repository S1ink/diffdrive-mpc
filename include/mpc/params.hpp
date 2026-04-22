#pragma once

namespace mpc
{

/// Central parameter struct — pass by value or const-ref throughout.
/// Every tunable number lives here so nothing is buried in .cpp files.
struct MPCParams
{
    // ── Horizon ──────────────────────────────────────────────────────
    int N = 25;        // prediction horizon (steps)
    double dt = 0.05;  // timestep [s]  → 20 Hz

    // ── Velocity limits ───────────────────────────────────────────────
    double v_max = 1.2;      // max forward speed   [m/s]
    double v_min = -0.1;      // min forward speed   [m/s]
    double omega_max = 1.2;  // max angular speed   [rad/s]

    // ── Acceleration limits ───────────────────────────────────────────
    double a_max = 1.0;      // linear accel bound  [m/s²]
    double alpha_max = 2.0;  // angular accel bound [rad/s²]

    // ── Reference cruise speed ────────────────────────────────────────
    double v_ref = 0.4;  // nominal forward speed [m/s]
    // actual speed per step is determined by look-ahead braking

    // ── Corridor / soft constraint ────────────────────────────────────
    double d_hard = 0.05;  // hard corridor half-width [m]
    //   = path_tolerance + noise_margin  (e.g. 5 cm + 3 cm)
    double w_slack = 2000.0;  // quadratic penalty on corridor slack ε_k

    // ── Tracking cost ─────────────────────────────────────────────────
    double Q_xy = 40.0;             // position weight (intermediate steps)
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
    double d_deadband = 0.025;  // ignore tracking errors below this [m]

    // ── Adaptive corridor ─────────────────────────────────────────────
    // When cross-track error exceeds d_hard, scale the corridor width up
    // by this factor so the solver remains feasible while recovering.
    double adaptive_corridor_scale = 1.8;

    // ── Initial Funneling ─────────────────────────────────────────────
    // If the robot starts outside the corridor, dynamically widen the
    // bounds at k=0 to swallow the error, then exponentially decay the
    // width back to d_hard over this many steps.
    double funnel_decay_tau = 5.0;

    // ── Velocity reduction under error ────────────────────────────────
    // v_ref_k *= clamp(1 − v_error_gain * |cte|,  v_min_scale, 1)
    double v_error_gain = 3.0;
    double v_min_scale = 0.0;  // 0 allows full stop for point-turn recovery

    // ── Reference blending ────────────────────────────────────────────
    // Smooths abrupt same-path numerical jitter: ref = α·new + (1−α)·old.
    // Blending is automatically suppressed on path identity changes.
    double blend_alpha = 0.7;  // 1.0 = no blending (pure new reference)

    // ── Stanley heading correction ────────────────────────────────────
    // Modifies the reference heading at each horizon step so the optimizer
    // is given a target that actively points the robot toward the path
    // during recovery, rather than always pointing along the path tangent.
    //
    // The correction is the Stanley steering formula:
    //   θ_ref_k += atan2(−stanley_k · cte,  max(v_k, stanley_v_min))
    //
    // When cte = 0 the correction vanishes (pure path-tangent reference).
    // As cte grows the heading target rotates toward the path, with the
    // correction saturating naturally at ±90° for large errors.
    // stanley_v_min prevents division-by-zero and bounds the correction
    // angle at low speed (effectively a maximum recovery yaw per metre).
    //
    // Recommended starting values: stanley_k ∈ [1.5, 4.0].
    // Larger values produce more aggressive rotation toward the path but
    // can over-correct at high speed; smaller values are gentler.
    double stanley_k = 2.5;
    double stanley_v_min = 0.15;  // [m/s]

    // ── Heading weight scaling ────────────────────────────────────────
    // Q_theta_eff = Q_theta * exp(−heading_scale_k * |cte|)
    //
    // IMPORTANT: set to 0.0 (disabled) unless you have a specific reason
    // to suppress heading cost at large CTE.  Suppressing heading weight
    // when the robot is far off-path prevents the rotation needed for
    // recovery and causes the robot to stop.  The Stanley correction above
    // is the correct tool for guiding the heading during recovery.
    double heading_scale_k = 0.0;

    // ── Path-change / projection reset ───────────────────────────────
    // Path changes are detected by hashing the full path geometry each
    // cycle.  This field is unused by the controller but kept for
    // reference / external tooling.
    double path_reset_threshold = 0.5;  // [m]  (legacy, not used)

    // ── Near-goal detection ───────────────────────────────────────────
    // When remaining path length drops below this, enforce v_N = 0.
    double goal_threshold = 0.03;  // [m]

    // ── Solver failure fallback ───────────────────────────────────────
    // On OSQP failure, return u_prev scaled by this factor.
    double fallback_decay = 0.8;
};

}  // namespace mpc