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
    int N = 24;
    double dt = 0.05;  // timestep [s]  → 20 Hz
    double feedback_delay_s = 0.0;

    // ── Velocity limits ───────────────────────────────────────────────
    double v_max = 1.2;      // max forward speed   [m/s]
    double v_min = -0.1;     // min forward speed   [m/s]
    double omega_max = 1.5;  // max angular speed   [rad/s]

    // ── Acceleration limits ───────────────────────────────────────────
    double a_max = 1.0;       // linear accel bound  [m/s²]
    double alpha_max = 5.0;  // angular accel bound [rad/s²]

    // ── Corridor / soft constraint ────────────────────────────────────
    double d_hard = 0.05;  // hard corridor half-width [m]
    //   = path_tolerance + noise_margin  (e.g. 5 cm + 3 cm)
    double w_slack = 100.0;  // quadratic penalty on corridor slack ε_k

    // ── Tracking cost ─────────────────────────────────────────────────
    double Q_xy = 20.0;             // position weight (intermediate steps)
    double Q_theta = 20.0;          // heading weight  (intermediate steps)
    double Q_xy_terminal = 50.0;    // elevated position weight at step N
    double Q_theta_terminal = 30.0;  // elevated heading weight  at step N

    // ── Control cost ─────────────────────────────────────────────────
    double R_v = 0.0;      // effort on v
    double R_omega = 0.0;  // effort on ω

    // ── Smoothness cost (penalises Δu between consecutive steps) ──────
    double R_rate_v = 1.0;      // weight on (v_k − v_{k-1})²
    double R_rate_omega = 1.0;  // weight on (ω_k − ω_{k-1})²

    // ── Initial Funneling ─────────────────────────────────────────────
    // If the robot starts outside the corridor, dynamically widen the
    // bounds at k=0 to swallow the error, then exponentially decay the
    // width back to d_hard over this many steps.
    double funnel_decay_tau = 5.0;

    // ── Reference blending ────────────────────────────────────────────
    // Smooths abrupt same-path numerical jitter: ref = α·new + (1−α)·old.
    // Blending is automatically suppressed on path identity changes.
    double blend_alpha = 1.0;  // 1.0 = no blending (pure new reference)

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
    double stanley_v_min = 0.15;   // [m/s]
    // Per-step exponential decay applied to the Stanley correction over the
    // horizon.  correction_k = correction_0 * exp(-stanley_decay * k).
    // At the default 0.15, correction drops to ~1% by step 30, so the
    // optimizer sees path-tangent headings at the far end of the horizon
    // while still being steered back toward the path at k=0.
    double stanley_decay = 0.15;

    // ── Near-goal detection ───────────────────────────────────────────
    // When remaining path length drops below this, enforce v_N = 0.
    double goal_threshold = 0.03;  // [m]

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
