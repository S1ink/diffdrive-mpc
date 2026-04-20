#include <iostream>
#include <vector>
#include <cmath>

#include "mpc/mpc_controller.hpp"
#include "mpc/types.hpp"
#include "mpc/path.hpp"

using namespace mpc;

// ── Plant model (nonlinear, ground truth) ─────────────────────────────────────
State plantStep(const State& x, const Control& u, double dt)
{
    return {
        x.x + u.v * std::cos(x.theta) * dt,
        x.y + u.v * std::sin(x.theta) * dt,
        x.theta + u.omega * dt};
}

// ── Straight-line path along the x-axis ──────────────────────────────────────
Path makeStraightPath(double x_start, double x_end, double y = 0.0)
{
    Path p;
    for (int i = 0; i <= 20; ++i)
    {
        const double t = (double)i / 20.0;
        PathPoint pt;
        pt.pos = Eigen::Vector2d(x_start + t * (x_end - x_start), y);
        p.pts.push_back(pt);
    }
    return p;
}

int main()
{
    // ── Parameters ────────────────────────────────────────────────────
    MPCParams p;
    p.N = 12;
    p.dt = 0.05;
    p.v_ref = 0.3;
    p.v_max = 0.5;
    p.v_min = 0.0;
    p.omega_max = 1.2;
    p.a_max = 0.4;
    p.alpha_max = 2.0;
    p.d_hard = 0.10;
    p.w_slack = 1500.0;
    p.Q_xy = 20.0;
    p.Q_theta = 2.0;
    p.Q_xy_terminal = 60.0;
    p.Q_theta_terminal = 6.0;
    p.R_rate_v = 2.0;
    p.R_rate_omega = 2.0;
    p.blend_alpha = 0.7;
    p.goal_threshold = 0.3;

    MPCController ctrl(p);

    // ── Initial robot state — starts laterally off the path ───────────
    State x;
    x.x = 0.0;
    x.y = 0.25;     // 25 cm lateral offset
    x.theta = 0.1;  // slight heading error

    Path path = makeStraightPath(0.0, 10.0, 0.0);

    std::cout << "t, x, y, theta, v, omega, cte\n";

    for (int t = 0; t < 300; ++t)
    {
        // At t = 150 simulate a path update (lateral offset of 0.5 m)
        // to test reference blending and projection reset.
        if (t == 150)
        {
            path = makeStraightPath(5.0, 15.0, 0.3);
            std::cout << "# --- path updated at t=150 ---\n";
        }

        const Control u = ctrl.update(x, path);

        // Cross-track error for logging (re-computed naively here)
        const Eigen::Vector2d pos(x.x, x.y);
        double best_d = 1e9;
        for (size_t i = 0; i < path.size() - 1; ++i)
        {
            const auto A = path.pts[i].pos;
            const auto AB = path.pts[i + 1].pos - A;
            const double t_seg =
                std::clamp((pos - A).dot(AB) / AB.squaredNorm(), 0.0, 1.0);
            best_d = std::min(best_d, (pos - (A + t_seg * AB)).norm());
        }

        x = plantStep(x, u, p.dt);

        std::cout << t << ", " << x.x << ", " << x.y << ", " << x.theta << ", "
                  << u.v << ", " << u.omega << ", " << best_d << "\n";
    }

    return 0;
}
