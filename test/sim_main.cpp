// =============================================================================
// sim_main.cpp  —  Differential-drive MPC simulation entry point
//
// All debug logging, noise injection, stuck detection, JSON formatting, and
// path helpers live in sim_utils.hpp / sim_utils.cpp.  This file is
// responsible only for:
//   1. CLI parsing and scenario / path selection.
//   2. The top-level simulation loop: update → log → step → terminate.
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "mpc/mpc_controller.hpp"
#include "mpc/params.hpp"
#include "mpc/path.hpp"
#include "mpc/types.hpp"
#include "sim_utils.hpp"

using namespace mpc;
using namespace mpc::sim;

int main(int argc, char** argv)
{
    // ── Controller parameters ─────────────────────────────────────────────
    MPCParams p;
    // Uncomment / tweak individual fields here as needed:
    // p.N         = 20;    // will be raised to minBrakingSteps() if smaller
    // p.dt        = 0.05;
    // p.v_ref     = 0.3;
    // p.v_max     = 0.5;
    // p.v_min     = 0.0;
    // p.omega_max = 1.5;
    // p.a_max     = 1.0;
    // p.alpha_max = 2.0;
    // p.d_hard    = 0.05;
    // p.w_slack   = 5000.0;
    // p.Q_xy      = 60.0;
    // p.Q_theta   = 30.0;
    // p.blend_alpha       = 0.7;
    // p.goal_threshold    = 0.3;

    MPCController ctrl(p);
    // p.N may have been raised by the controller — use ctrl.debugInfo() or
    // re-read p after construction if you need the effective value.

    // ── Simulation termination limits ─────────────────────────────────────
    static constexpr int MAX_STEPS = 10000;

    // ── CLI parsing ───────────────────────────────────────────────────────
    int scenario = 0;
    std::string custom_path_file;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--scenario" && i + 1 < argc)
        {
            scenario = std::stoi(argv[++i]);
        }
        if (arg == "--path" && i + 1 < argc)
        {
            custom_path_file = argv[++i];
        }
    }

    // ── Initial robot state and path ──────────────────────────────────────
    State x = {0.0, 0.25, 0.1};
    Path path;

    if (!custom_path_file.empty())
    {
        path = loadPathFromFile(custom_path_file);

        if (path.pts.size() >= 2)
        {
            x.x = path.pts[0].pos.x();
            x.y = path.pts[0].pos.y();
            x.theta = std::atan2(
                path.pts[1].pos.y() - path.pts[0].pos.y(),
                path.pts[1].pos.x() - path.pts[0].pos.x());
            std::cerr << "[sim] Snapped robot to custom path start: (" << x.x
                      << ", " << x.y << ", " << x.theta << " rad)\n";
        }
    }
    else
    {
        switch (scenario)
        {
            case 1:
                path = makeSharpLPath();
                break;
            case 2:
                path = makeFigure8Path();
                x = {0.0, 0.5, 0.0};
                break;
            case 3:
                path = makeStraightPath(0, 10);
                x = {-2.0, 3.0, M_PI};
                break;
            default:
                path = makeStraightPath(0.0, 10.0, 0.0);
                break;
        }
    }

    // ── Testing utilities ─────────────────────────────────────────────────
    StateNoise noise;
    StuckDetector stuck;
    FrameLogger logger("sim_data.jsonl");

    logger.logParams(p, MAX_STEPS);

    // ── Simulation loop ───────────────────────────────────────────────────
    for (int step = 0; step < MAX_STEPS; ++step)
    {
        const double sim_t = step * p.dt;

        // Dynamic path update for the default scenario (scenario 0).
        if (scenario == 0 && custom_path_file.empty() && step == 150)
        {
            path = makeStraightPath(10.0, 15.0, 0.3);
            std::cerr << "[sim] Path updated at step " << step
                      << " (t=" << sim_t << "s)\n";
        }

        // Add sensor noise before passing state to the controller.
        const State x_noisy = noise.apply(x);

        const Control u = ctrl.update(x_noisy, path);

        // Optionally prune traversed path segments (uncomment to enable):
        const size_t pruned = ctrl.pruneTraversedSegments(path);
        if (pruned > 0)
            std::cerr << "[sim] t=" << sim_t << "s: pruned " << pruned
                      << " segment(s), path now " << path.size() << " pts\n";

        const DebugInfo& dbg = ctrl.debugInfo();
        logger.logFrame(sim_t, x_noisy, u, path, dbg);

        // Advance ground-truth state with the clean (noiseless) plant model.
        x = plantStep(x, u, p.dt);

        // ── Termination: goal reached ──────────────────────────────────
        if (dbg.near_goal)
        {
            std::cerr << "[sim] Goal reached at t=" << sim_t + p.dt
                      << "s (step " << step + 1 << ")\n";
            break;
        }

        // ── Termination: robot stuck ───────────────────────────────────
        stuck.update(x);
        if (stuck.isStuck(p.dt))
        {
            std::cerr << "[sim] Robot stuck at t=" << sim_t + p.dt << "s\n";
            break;
        }
    }

    return 0;
}
