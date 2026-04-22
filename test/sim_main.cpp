// =============================================================================
// sim_main.cpp  —  Differential-drive MPC simulation
// =============================================================================

#include <cmath>
#include <fstream>
#include <utility>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>

#include "mpc/mpc_controller.hpp"
#include "mpc/path.hpp"
#include "mpc/types.hpp"

using namespace mpc;

// ── Plant model (nonlinear ground truth) ─────────────────────────────────────
static State plantStep(const State& x, const Control& u, double dt)
{
    return {
        x.x + u.v * std::cos(x.theta) * dt,
        x.y + u.v * std::sin(x.theta) * dt,
        x.theta + u.omega * dt};
}

// ── Path factories ───────────────────────────────────────────────────────────
static Path
    makeStraightPath(double x_start, double x_end, double y = 0.0, int pts = 20)
{
    Path p;
    for (int i = 0; i <= pts; ++i)
    {
        const double t = (double)i / (double)pts;
        p.pts.push_back({Eigen::Vector2d(x_start + t * (x_end - x_start), y)});
    }
    return p;
}

static Path makeSharpLPath()
{
    Path p;
    p.pts.push_back({Eigen::Vector2d(0, 0)});
    p.pts.push_back({Eigen::Vector2d(1, 0)});
    p.pts.push_back({Eigen::Vector2d(2, 0)});
    p.pts.push_back(
        {Eigen::Vector2d(2, 0.1)});  // helps projection wrap the corner
    p.pts.push_back({Eigen::Vector2d(2, 1)});
    p.pts.push_back({Eigen::Vector2d(2, 2)});
    return p;
}

static Path makeFigure8Path(int n = 100, double scale = 3.0)
{
    Path p;
    for (int i = 0; i <= n; ++i)
    {
        double t = 2.0 * M_PI * i / n;
        p.pts.push_back({Eigen::Vector2d(
            scale * std::sin(t),
            scale * std::sin(t) * std::cos(t))});
    }
    return p;
}

static Path loadPathFromFile(const std::string& filepath)
{
    Path p;
    std::ifstream file(filepath);
    double x, y;
    while (file >> x >> y)
    {
        p.pts.push_back({Eigen::Vector2d(x, y)});
    }
    return p;
}

// ── JSON helpers ─────────────────────────────────────────────────────────────
static void jd(std::ostream& s, double v, int prec = 5)
{
    s << std::fixed << std::setprecision(prec) << v;
}

// Bug #1 fix: returns a string so callers route through a single emitLine()
//   lambda that writes to both stdout and data_file — the old two-stream
//   design silently discarded every write to out2.
// Bug #2 fix: sim_t is floating-point seconds (step * dt) so the Python
//   visualiser's time axis is correct; the old int step index was off by 20×.
static std::string buildFrame(
    double sim_t,
    const State& x,
    const Control& u,
    const Path& path,
    const DebugInfo& dbg)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "{";
    ss << "\"t\":";
    jd(ss, sim_t, 4);
    ss << ",";
    ss << "\"robot\":{\"x\":" << x.x << ",\"y\":" << x.y
       << ",\"theta\":" << x.theta << "},";
    ss << "\"control\":{\"v\":" << u.v << ",\"omega\":" << u.omega << "},";
    ss << "\"cte\":";
    jd(ss, dbg.cte_raw);
    ss << ",";
    ss << "\"proj\":[";
    jd(ss, dbg.proj_pt.x());
    ss << ",";
    jd(ss, dbg.proj_pt.y());
    ss << "],";
    ss << "\"d_hard_eff\":";
    jd(ss, dbg.d_hard_eff);
    ss << ",";
    ss << "\"solver_ok\":" << (dbg.solver_ok ? "true" : "false") << ",";
    ss << "\"solve_ms\":";
    jd(ss, dbg.solve_ms, 4);
    ss << ",";

    ss << "\"pred\":[";
    for (int k = 0; k < (int)dbg.pred_traj.size(); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[" << dbg.pred_traj[k].x << "," << dbg.pred_traj[k].y << ","
           << dbg.pred_traj[k].theta << "]";
    }
    ss << "],\"ref\":[";
    for (int k = 0; k < (int)dbg.ref_traj.size(); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[" << dbg.ref_traj[k].x << "," << dbg.ref_traj[k].y << ","
           << dbg.ref_traj[k].theta << "]";
    }
    ss << "],\"path\":[";
    for (int i = 0; i < (int)path.size(); ++i)
    {
        if (i)
        {
            ss << ",";
        }
        ss << "[" << path.pts[i].pos.x() << "," << path.pts[i].pos.y() << "]";
    }
    ss << "]}";
    return ss.str();
}

// ── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    MPCParams p;  // Assuming default params set in your struct
    // p.N = 20;
    // p.dt = 0.05;
    // p.v_ref = 0.3;
    // p.v_max = 0.5;
    // p.v_min = 0.0;
    // p.omega_max = 1.5;
    // p.a_max = 1.0;
    // p.alpha_max = 2.0;
    // p.d_hard = 0.05;
    // p.adaptive_corridor_scale = 3.0;
    // p.funnel_decay_tau = 10.0;
    // p.v_min_scale = 0.2;
    // p.v_error_gain = 5.0;
    // p.w_slack = 5000.0;
    // p.Q_xy = 60.0;
    // p.Q_theta = 30.0;
    // p.Q_xy_terminal = 60.0;
    // p.Q_theta_terminal = 100.0;
    // p.R_v = 5.0;
    // p.R_omega = 5.0;
    // p.R_rate_v = 2.0;
    // p.R_rate_omega = 3.0;
    // p.heading_scale_k = 1.0;
    // p.blend_alpha = 0.7;
    // p.goal_threshold = 0.3;

    MPCController ctrl(p);

    std::mt19937 gen{std::random_device{}()};
    std::normal_distribution pos_dist(0.0, 0.01);
    std::normal_distribution ang_dist(0.0, 0.0085);

    int scenario = 0;
    std::string custom_path_file = "";

    // Basic CLI parsing
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--scenario" && i + 1 < argc)
        {
            scenario = std::stoi(argv[++i]);
        }
        if (arg == "--path" && i + 1 < argc)
        {
            custom_path_file = argv[++i];
        }
    }

    State x = {0.0, 0.25, 0.1};  // Default offset
    Path path;

    if (!custom_path_file.empty())
    {
        path = loadPathFromFile(custom_path_file);

        // ── AUTO-INITIALIZE ROBOT POSE ────────────────────────────
        // If a custom path is loaded, teleport the robot to the start
        // so the initial CTE is 0 and the solver doesn't panic.
        if (path.pts.size() >= 2)
        {
            x.x = path.pts[0].pos.x();
            x.y = path.pts[0].pos.y();
            // Point the robot toward the second waypoint
            x.theta = std::atan2(
                path.pts[1].pos.y() - path.pts[0].pos.y(),
                path.pts[1].pos.x() - path.pts[0].pos.x());
            std::cerr << "[sim] Snapped robot to custom path start: "
                      << "(" << x.x << ", " << x.y << ", " << x.theta
                      << " rad)\n";
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
                break;  // Poor init
            default:
                path = makeStraightPath(0.0, 10.0, 0.0);
                break;
        }
    }

    std::ofstream data_file("sim_data.jsonl");
    auto emitLine = [&](const std::string& s)
    {
        std::cout << s << "\n";
        if (data_file.is_open())
        {
            data_file << s << "\n";
        }
    };

    // ── Termination parameters ────────────────────────────────────────────────
    // Run until:
    //  (a) controller reports near_goal  (remaining arc < goal_threshold), OR
    //  (b) robot is stuck: net displacement over STUCK_WINDOW steps < STUCK_DIST_M
    //      AND average forward speed < STUCK_SPEED_MPS, OR
    //  (c) MAX_STEPS hard safety cap.
    static constexpr int MAX_STEPS = 10000;
    static constexpr int STUCK_WINDOW = 100;  // steps  (~5 s at 20 Hz)
    static constexpr double STUCK_DIST_M =
        0.05;  // m  net displacement threshold
    static constexpr double STUCK_SPEED_MPS =
        0.02;  // m/s average speed threshold

    std::ostringstream hdr;
    hdr << "{\"params\":{\"N\":" << p.N << ",\"dt\":" << p.dt
        << ",\"v_ref\":" << p.v_ref << ",\"v_max\":" << p.v_max
        << ",\"omega_max\":" << p.omega_max << ",\"d_hard\":" << p.d_hard
        << ",\"max_steps\":" << MAX_STEPS << "}}";
    emitLine(hdr.str());

    // Sliding window of positions for stuck detection.
    std::vector<std::pair<double, double>> pos_history;
    pos_history.reserve(STUCK_WINDOW + 1);

    for (int step = 0; step < MAX_STEPS; ++step)
    {
        const double sim_t = step * p.dt;

        // Only trigger the dynamic path update if we are in the default scenario
        // AND we haven't loaded a custom path.
        if (scenario == 0 && custom_path_file.empty() && step == 150)
        {
            path = makeStraightPath(3.0, 5.0, 0.3);
            std::cerr << "[sim] path updated at step " << step
                      << " (t=" << sim_t << "s)\n";
        }

        State x2 = x;
        x2.x += pos_dist(gen);
        x2.y += pos_dist(gen);
        x2.theta += ang_dist(gen);
        const Control u = ctrl.update(x2, path);
        // const size_t pruned = ctrl.pruneTraversedSegments(path);
        // if (pruned > 0)
        // {
        //     std::cerr << "[sim] t=" << sim_t << "s: pruned " << pruned
        //               << " segment(s), path now " << path.size() << " pts\n";
        // }
        const DebugInfo& dbg = ctrl.debugInfo();

        emitLine(buildFrame(sim_t, x2, u, path, dbg));
        std::cout.flush();

        x = plantStep(x, u, p.dt);

        // ── (a) Goal reached ───────────────────────────────────────────────────
        if (dbg.near_goal)
        {
            std::cerr << "[sim] Goal reached at t=" << sim_t + p.dt
                      << "s (step " << step + 1 << ")\n";
            break;
        }

        // ── (b) Stuck detection ────────────────────────────────────────────────
        pos_history.push_back({x.x, x.y});
        if ((int)pos_history.size() > STUCK_WINDOW)
        {
            pos_history.erase(pos_history.begin());
        }

        if ((int)pos_history.size() == STUCK_WINDOW)
        {
            const double dx = x.x - pos_history.front().first;
            const double dy = x.y - pos_history.front().second;
            const double net_disp = std::sqrt(dx * dx + dy * dy);

            double path_len = 0.0;
            for (int j = 1; j < STUCK_WINDOW; ++j)
            {
                const double ddx =
                    pos_history[j].first - pos_history[j - 1].first;
                const double ddy =
                    pos_history[j].second - pos_history[j - 1].second;
                path_len += std::sqrt(ddx * ddx + ddy * ddy);
            }
            const double avg_speed = path_len / (STUCK_WINDOW * p.dt);

            if (net_disp < STUCK_DIST_M && avg_speed < STUCK_SPEED_MPS)
            {
                std::cerr << "[sim] Robot stuck at t=" << sim_t + p.dt
                          << "s (net_disp=" << net_disp
                          << "m, avg_speed=" << avg_speed << "m/s)\n";
                break;
            }
        }
    }
    return 0;
}
