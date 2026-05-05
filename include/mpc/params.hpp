#pragma once

#include <cmath>


namespace mpc
{

// Central parameter struct. Tunable values for MPC behaviour.
struct MPCParams
{
    // Prediction horizon
    int N = 24;
    double dt = 0.05;  // timestep [s]
    double feedback_delay_s = 0.0;

    // Velocity limits
    double v_max = 2.5;
    double v_min = -0.1;
    double omega_max = 1.5;

    // Acceleration limits
    double a_max = 1.0;
    double alpha_max = 5.0;

    // Corridor / slack
    double d_hard = 0.05;    // hard corridor half-width [m]
    double w_slack = 100.0;  // quadratic penalty on slack eps_k

    // Tracking / control costs
    double Q_xy = 20.0;
    double Q_theta = 20.0;
    double Q_xy_terminal = 50.0;
    double Q_theta_terminal = 30.0;
    double R_v = 0.0;
    double R_omega = 0.0;
    double R_rate_v = 3.0;
    double R_rate_omega = 3.0;

    // Funnel decay: how quickly an initial widened corridor returns to d_hard
    double funnel_decay_tau = 5.0;

    // Reference blending (alpha = 1.0 uses new reference only)
    double blend_alpha = 1.0;

    // Stanley heading correction parameters
    double stanley_k = 2.5;
    double stanley_v_min = 0.15;
    double stanley_decay = 0.15;

    // Near-goal detection
    double goal_threshold = 0.03;
    double goal_cte_scale = 2.0;
    double goal_stop_vel = 0.01;

    // Solver fallback multiplier on failure
    double fallback_decay = 0.8;

    // Minimum steps to brake from v_max to 0 under a_max
    int minBrakingSteps() const
    {
        const double dv = v_max - std::max(v_min, 0.0);
        return static_cast<int>(std::ceil(dv / (a_max * dt)));
    }
};

}  // namespace mpc
