#pragma once

#include <cmath>


namespace mpc
{

/// Central parameter struct — pass by value or const-ref throughout.
/// Every tunable number lives here so nothing is buried in .cpp files.
struct MPCParams
{
    // ── Horizon ──────────────────────────────────────────────────────
    //
    // N is the number of prediction steps.  MPCController enforces at
    // construction time that N >= minBrakingSteps() so that the full
    // maximum-braking distance (v_max → 0 under a_max) is always within
    // the horizon.  This guarantees that every upcoming corner speed
    // limit is visible to the look-ahead braking integrator in the
    // reference generator.
    //
    // With the defaults (v_max=1.5, a_max=1.0, dt=0.05) minBrakingSteps()
    // returns 30, so the effective N is raised from the nominal 15 to 30.
    // Override N here if you want a longer horizon; it will never be
    // silently shortened below minBrakingSteps().
    int N =
        15;  // prediction horizon (steps) — raised to minBrakingSteps() if smaller
    double dt = 0.05;  // timestep [s]  → 20 Hz

    // ── Velocity limits ───────────────────────────────────────────────
    double v_max = 1.2;      // max forward speed   [m/s]
    double v_min = -0.1;     // min forward speed   [m/s]
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
    double w_slack = 100.0;  // quadratic penalty on corridor slack ε_k

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
    double d_deadband = 0.015;  // ignore tracking errors below this [m]

    // ── Adaptive corridor ─────────────────────────────────────────────
    // When cross-track error exceeds d_hard, scale the corridor width up
    // by this factor so the solver remains feasible while recovering.
    double adaptive_corridor_scale = 1.3;

    // ── Initial Funneling ─────────────────────────────────────────────
    // If the robot starts outside the corridor, dynamically widen the
    // bounds at k=0 to swallow the error, then exponentially decay the
    // width back to d_hard over this many steps.
    double funnel_decay_tau = 5.0;

    // ── Velocity reduction under error ────────────────────────────────
    // v_ref_k *= clamp(1 − v_error_gain * |cte|,  v_min_scale, 1)
    double v_error_gain = 3.0;
    double v_min_scale = 0.1;  // 0 allows full stop for point-turn recovery

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
    double stanley_k = 3.5;
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

    // ── Recovery ─────────────────────────────────────────────────────────
    // When the corrected reference heading at k=0 differs from the robot's
    // current heading by more than this angle, the controller bypasses the QP
    // entirely and issues a pure point-turn (v=0, omega=±omega_max) until the
    // heading is within threshold.  This handles tight U-turns and the
    // "robot facing entirely wrong direction" case that OSQP cannot resolve
    // through gradient descent alone.
    //
    // Set to M_PI (180°) to disable.  100° is a good starting value.
    double recovery_heading_threshold = 100.0 * M_PI / 180.0;  // [rad]

    // ── nearGoal CTE gate ─────────────────────────────────────────────────
    // The terminal-velocity-zero constraint is only applied when the robot is
    // both close to the path end (arc < goal_threshold) AND laterally close
    // to the path centre (|cte| < goal_cte_scale * d_hard).
    // Without this gate the robot can get frozen at the path endpoint while
    // still displaced sideways (e.g. when a path update puts the endpoint
    // behind the robot).  Default 2.0 = allow terminal stop up to 2×d_hard.
    double goal_cte_scale = 2.0;

    // ── Solver failure fallback ───────────────────────────────────────────
    // On OSQP failure, return u_prev scaled by this factor.
    double fallback_decay = 0.8;

    // ── Derived / validation ──────────────────────────────────────────────

    /// Minimum prediction horizon steps required to observe the complete
    /// braking event from v_max down to rest under maximum deceleration.
    ///
    /// Formula:  ceil( (v_max − max(v_min, 0)) / (a_max · dt) )
    ///
    /// MPCController enforces N >= minBrakingSteps() in its constructor so
    /// that the look-ahead braking integrator inside ReferenceGenerator can
    /// always "see" every upcoming velocity limit before the robot needs to
    /// start decelerating for it.
    int minBrakingSteps() const
    {
        const double dv = v_max - std::max(v_min, 0.0);
        return static_cast<int>(std::ceil(dv / (a_max * dt)));
    }
};

}  // namespace mpc
