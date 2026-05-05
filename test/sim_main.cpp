// =============================================================================
// sim_main.cpp - Differential-drive MPC simulation entry point
//
// All debug logging, noise injection, stuck detection, JSON formatting, and
// path helpers live in sim_utils.hpp / sim_utils.cpp.  This file is
// responsible only for:
//   1. CLI parsing and scenario / path selection.
//   2. The top-level simulation loop: update -> log -> step -> terminate.
// =============================================================================

#include "mpc/path.hpp"
#include "mpc/types.hpp"
#include "sim_utils.hpp"
#include "mpc/params.hpp"
#include "mpc/mpc_controller.hpp"

#include <cmath>
#include <chrono>
#include <string>
#include <iostream>

using namespace mpc;
using namespace mpc::sim;


int main(int argc, char** argv)
{
    // Simulation termination limits
    static constexpr int MAX_STEPS = 10000;

    // CLI parsing
    int scenario = 0;
    std::string custom_path_file;
    std::string config_file;
    bool noisy = false;
    bool prune = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--scenario" && i + 1 < argc)
        {
            scenario = std::stoi(argv[++i]);
        }
        else if (arg == "--path" && i + 1 < argc)
        {
            custom_path_file = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc)
        {
            config_file = argv[++i];
        }
        else if (arg == "--noisy")
        {
            noisy = true;
        }
        else if (arg == "--prune")
        {
            prune = true;
        }
    }

    // Load controller parameters (from file if provided, otherwise defaults)
    MPCParams p;
    if (!config_file.empty())
    {
        try
        {
            p = loadParamsFromFile(config_file);
            std::cerr << "[sim] Loaded MPC params from: " << config_file
                      << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[sim] Error loading config file: " << e.what()
                      << " -- using built-in defaults.\n";
        }
    }

    MPCController ctrl(p);

    // Initial robot state and path
    State x;
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
            case 0:
                x = {0.0, 0.0, M_PI};
                path = makeStraightPath(0.0, 10.0, 0.0);
                break;
            case 1:
                x = {0.0, 0.25, 0.1};
                path = makeSharpLPath();
                break;
            case 2:
                x = {0.0, 0.5, 0.0};
                path = makeFigure8Path();
                break;
            case 3:
                x = {-2.0, 3.0, M_PI};
                path = makeStraightPath(0, 10);
                break;
            default:
                x = {0.0, 0.0, 0.0};
                path = makeStraightPath(0.0, 10.0, 0.0);
                break;
        }
    }

    // Testing utilities
    StateNoise noise;
    StuckDetector stuck;
    FrameLogger logger("sim_data.jsonl");

    logger.logParams(p, MAX_STEPS);

    // Simulation loop
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    for (int step = 0; step < MAX_STEPS; ++step)
    {
        const double sim_t = step * p.dt;

        // Dynamic path update for the default scenario (scenario 0).
        // if (scenario == 0 && custom_path_file.empty() && step == 150)
        // {
        //     path = makeStraightPath(10.0, 15.0, 0.3);
        //     std::cerr << "[sim] Path updated at step " << step
        //               << " (t=" << sim_t << "s)\n";
        // }

        // Add sensor noise before passing state to the controller if enabled.
        const State x_noisy = noisy ? noise.apply(x) : x;

        const Control u = ctrl.update(x_noisy, path);

        // Prune traversed path segments when --prune is active.
        if (prune)
        {
            const size_t pruned = ctrl.pruneTraversedSegments(path);
            if (pruned > 0)
            {
                std::cerr << "[sim] t=" << sim_t << "s: pruned " << pruned
                          << " segment(s), path now " << path.size()
                          << " pts\n";
            }
        }

        const DebugInfo& dbg = ctrl.debugInfo();
        logger.logFrame(sim_t, x_noisy, u, path, dbg);

        // Advance ground-truth state with the clean (noiseless) plant model.
        x = plantStep(x, u, p.dt);

        if (dbg.near_goal && std::abs(u.v) < p.goal_stop_vel)
        {
            std::cerr << "[sim] Goal reached at t=" << sim_t + p.dt
                      << "s (step " << step + 1 << ")"
                      << "  v=" << u.v << " m/s\n";
            break;
        }

        // Termination: robot stuck
        stuck.update(x);
        if (stuck.isStuck(p.dt))
        {
            std::cerr << "[sim] Robot stuck at t=" << sim_t + p.dt << "s\n";
            break;
        }
    }

    std::cerr << "[sim] Sim completed in "
              << (std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - start))
                     .count()
              << "s\n";

    return 0;
}
