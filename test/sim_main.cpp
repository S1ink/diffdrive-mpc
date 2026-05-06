// =============================================================================
// sim_main.cpp - Differential-drive MPC simulation entry point
//
// Modes
// -----
//   (default)            Standard single-path run.
//   --scenario N         One of the built-in named scenarios.
//   --path file.txt      Custom whitespace-delimited "x y" path file.
//   --random             Random staged mode: the robot starts at a random pose
//                        and follows a randomly generated path.  When it
//                        approaches the end of the current path a new one is
//                        seamlessly generated from the robot's current pose,
//                        exercising the MPC's dynamic path-update handling.
//   --complexity 1-5     Governs path difficulty in --random mode (default 2):
//                          1  gentle curves, short segments (4 segs x 1.5 m)
//                          3  moderate turns, medium length (8 segs x 2.5 m)
//                          5  sharp turns, long segments   (12 segs x 3.5 m)
//   --stages N           Number of path stages before stopping (default 5).
//   --seed N             RNG seed for reproducible random runs.
//   --config file.cfg    Load MPCParams from a key=value config file.
//   --noisy              Apply Gaussian sensor noise to the state estimate.
//   --prune              Prune traversed path segments each step.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Distance from the robot to the last point of the path.
static double distToPathEnd(const State& x, const Path& path)
{
    if (path.pts.empty())
    {
        return 0.0;
    }
    const Eigen::Vector2d& end = path.pts.back().pos;
    const double dx = x.x - end.x();
    const double dy = x.y - end.y();
    return std::sqrt(dx * dx + dy * dy);
}

// Heading of the last path segment, used to seed the next stage so the robot
// never faces a sudden U-turn at a stage boundary.
static double pathEndHeading(const Path& path)
{
    const int n = static_cast<int>(path.pts.size());
    if (n < 2)
    {
        return 0.0;
    }
    const Eigen::Vector2d d = path.pts[n - 1].pos - path.pts[n - 2].pos;
    return std::atan2(d.y(), d.x());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    static constexpr int MAX_STEPS_PER_STAGE = 5000;
    static constexpr int MAX_STEPS_NORMAL = 10000;

    // ---- CLI parsing -------------------------------------------------------
    int scenario = 0;
    int complexity = 2;
    int n_stages = 5;
    bool random_mode = false;
    bool noisy = false;
    bool prune = false;
    bool seed_set = false;
    unsigned int rng_seed = 0;
    std::string custom_path_file;
    std::string config_file;

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
        else if (arg == "--complexity" && i + 1 < argc)
        {
            complexity = std::stoi(argv[++i]);
        }
        else if (arg == "--stages" && i + 1 < argc)
        {
            n_stages = std::stoi(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            rng_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            seed_set = true;
        }
        else if (arg == "--random")
        {
            random_mode = true;
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

    complexity = std::max(1, std::min(5, complexity));

    // ---- MPC params --------------------------------------------------------
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
            std::cerr << "[sim] Error loading config: " << e.what()
                      << " -- using built-in defaults.\n";
        }
    }

    MPCController ctrl(p);

    // ---- RNG ---------------------------------------------------------------
    if (!seed_set)
    {
        rng_seed = std::random_device{}();
    }
    std::mt19937 rng(rng_seed);
    std::cerr << "[sim] RNG seed: " << rng_seed << "\n";

    // ---- Utilities ---------------------------------------------------------
    StateNoise noise;
    StuckDetector stuck;
    FrameLogger logger("sim_data.jsonl");

    const int total_max_steps =
        random_mode ? n_stages * MAX_STEPS_PER_STAGE : MAX_STEPS_NORMAL;

    logger.logParams(p, total_max_steps);

    // =========================================================================
    // RANDOM STAGED MODE
    // =========================================================================
    if (random_mode)
    {
        std::cerr << "[sim] Random mode: complexity=" << complexity
                  << "  stages=" << n_stages << "\n";

        // Random initial pose (small offset from origin, any heading).
        std::uniform_real_distribution<double> pos_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> hdg_dist(-M_PI, M_PI);

        State x = {pos_dist(rng), pos_dist(rng), hdg_dist(rng)};
        std::cerr << "[sim] Initial pose: (" << x.x << ", " << x.y << ", "
                  << x.theta * 180.0 / M_PI << " deg)\n";

        // First stage path, seeded from the initial pose so the robot is
        // already facing roughly the right direction.
        Path path = makeRandomPath(x.x, x.y, x.theta, complexity, rng);
        std::cerr << "[sim] Stage 1/" << n_stages << ": " << path.size()
                  << " pts\n";

        // Trigger distance: how close to the path end before we swap.
        // Larger at higher complexity (faster, longer paths need more lead).
        const double stage_trigger_dist = 0.5 + complexity * 0.3;

        int stage = 1;
        int step = 0;
        bool sim_active = true;

        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();

        while (sim_active && step < total_max_steps)
        {
            const double sim_t = step * p.dt;

            const State x_noisy = noisy ? noise.apply(x) : x;
            const Control u = ctrl.update(x_noisy, path);

            if (prune)
            {
                const size_t pruned = ctrl.pruneTraversedSegments(path);
                if (pruned > 0)
                {
                    std::cerr << "[sim] t=" << sim_t << "s: pruned " << pruned
                              << " segment(s)\n";
                }
            }

            const DebugInfo& dbg = ctrl.debugInfo();
            logger.logFrame(sim_t, x_noisy, u, path, dbg);

            x = plantStep(x, u, p.dt);

            // ---- Stage transition check ------------------------------------
            if (distToPathEnd(x, path) < stage_trigger_dist)
            {
                if (stage >= n_stages)
                {
                    // Final stage: wait for the robot to actually stop.
                    const double v_stop = p.v_max * 0.05;
                    if (dbg.near_goal && std::abs(u.v) < v_stop)
                    {
                        std::cerr << "[sim] All " << n_stages
                                  << " stages complete at t=" << sim_t + p.dt
                                  << "s\n";
                        sim_active = false;
                    }
                }
                else
                {
                    // Generate next stage from current pose, using the
                    // outgoing path heading so there is no heading jump.
                    ++stage;
                    const double end_hdg = pathEndHeading(path);
                    path = makeRandomPath(x.x, x.y, end_hdg, complexity, rng);
                    std::cerr << "[sim] Stage " << stage << "/" << n_stages
                              << " at t=" << sim_t + p.dt << "s"
                              << "  pts=" << path.size() << "\n";

                    // Clear the stuck window so the slow approach speed at
                    // the end of a stage does not poison the next one.
                    stuck = StuckDetector{};
                }
            }

            // ---- Stuck check -----------------------------------------------
            stuck.update(x);
            if (stuck.isStuck(p.dt))
            {
                std::cerr << "[sim] Robot stuck at t=" << sim_t + p.dt
                          << "s (stage " << stage << "/" << n_stages << ")\n";
                sim_active = false;
            }

            ++step;
        }

        if (step >= total_max_steps)
        {
            std::cerr << "[sim] Max steps (" << total_max_steps
                      << ") reached.\n";
        }

        std::cerr << "[sim] Sim completed in "
                  << (std::chrono::duration<double>(
                          std::chrono::high_resolution_clock::now() - start))
                         .count()
                  << "s\n";

        return 0;
    }

    // =========================================================================
    // STANDARD MODE
    // =========================================================================

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
            std::cerr << "[sim] Snapped to path start: (" << x.x << ", " << x.y
                      << ", " << x.theta << " rad)\n";
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

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    for (int step = 0; step < MAX_STEPS_NORMAL; ++step)
    {
        const double sim_t = step * p.dt;

        // Dynamic path update for the default scenario (scenario 0).
        // if (scenario == 0 && custom_path_file.empty() && step == 150)
        // {
        //     path = makeStraightPath(10.0, 15.0, 0.3);
        //     std::cerr << "[sim] Path updated at step " << step
        //               << " (t=" << sim_t << "s)\n";
        // }

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
