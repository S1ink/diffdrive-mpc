// =============================================================================
// sim_main.cpp  —  Differential-drive MPC simulation
//
// Output format:  JSONL (one JSON object per line) written to both stdout
// and "sim_data.jsonl" in the working directory.
//
// First line: a params object so the visualizer can set axis limits.
// Subsequent lines: one frame object per control cycle.
//
// Frame schema:
//   {
//     "t":          int,          // step index
//     "robot":      {x,y,theta},
//     "control":    {v,omega},
//     "cte":        float,        // raw (un-deadbanded) signed cross-track error
//     "proj":       [x,y],        // nearest point on path
//     "d_hard_eff": float,        // effective corridor half-width this cycle
//     "solver_ok":  bool,
//     "pred":       [[x,y,th],…], // MPC predicted horizon x_0..x_N
//     "ref":        [[x,y,th],…], // blended reference    x_0..x_N
//     "path":       [[x,y],…]     // current path polyline
//   }
// =============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

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

// ── Path factory ─────────────────────────────────────────────────────────────

static Path
    makeStraightPath(double x_start, double x_end, double y = 0.0, int pts = 20)
{
    Path p;
    for (int i = 0; i <= pts; ++i)
    {
        const double t = (double)i / (double)pts;
        PathPoint pt;
        pt.pos = Eigen::Vector2d(x_start + t * (x_end - x_start), y);
        p.pts.push_back(pt);
    }
    return p;
}

// ── JSON helpers ─────────────────────────────────────────────────────────────

/// Write a double with fixed precision to avoid scientific notation.
static void jd(std::ostream& s, double v, int prec = 5)
{
    s << std::fixed << std::setprecision(prec) << v;
}

/// Emit one complete frame as a single JSON line to both output streams.
static void writeFrame(
    std::ostream& out1,
    std::ostream& out2,
    int t,
    const State& x,
    const Control& u,
    const Path& path,
    const DebugInfo& dbg)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);

    ss << "{";
    ss << "\"t\":" << t << ",";

    // Robot state
    ss << "\"robot\":{\"x\":";
    jd(ss, x.x);
    ss << ",\"y\":";
    jd(ss, x.y);
    ss << ",\"theta\":";
    jd(ss, x.theta);
    ss << "},";

    // Control
    ss << "\"control\":{\"v\":";
    jd(ss, u.v);
    ss << ",\"omega\":";
    jd(ss, u.omega);
    ss << "},";

    // CTE (raw, signed)
    ss << "\"cte\":";
    jd(ss, dbg.cte_raw);
    ss << ",";

    // Projection point
    ss << "\"proj\":[";
    jd(ss, dbg.proj_pt.x());
    ss << ",";
    jd(ss, dbg.proj_pt.y());
    ss << "],";

    // Effective corridor half-width
    ss << "\"d_hard_eff\":";
    jd(ss, dbg.d_hard_eff);
    ss << ",";

    // Solver status
    ss << "\"solver_ok\":" << (dbg.solver_ok ? "true" : "false") << ",";

    // Predicted horizon
    ss << "\"pred\":[";
    for (int k = 0; k < (int)dbg.pred_traj.size(); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[";
        jd(ss, dbg.pred_traj[k].x);
        ss << ",";
        jd(ss, dbg.pred_traj[k].y);
        ss << ",";
        jd(ss, dbg.pred_traj[k].theta);
        ss << "]";
    }
    ss << "],";

    // Reference horizon
    ss << "\"ref\":[";
    for (int k = 0; k < (int)dbg.ref_traj.size(); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[";
        jd(ss, dbg.ref_traj[k].x);
        ss << ",";
        jd(ss, dbg.ref_traj[k].y);
        ss << ",";
        jd(ss, dbg.ref_traj[k].theta);
        ss << "]";
    }
    ss << "],";

    // Current path polyline
    ss << "\"path\":[";
    for (int i = 0; i < (int)path.size(); ++i)
    {
        if (i)
        {
            ss << ",";
        }
        ss << "[";
        jd(ss, path.pts[i].pos.x());
        ss << ",";
        jd(ss, path.pts[i].pos.y());
        ss << "]";
    }
    ss << "]";

    ss << "}\n";

    const std::string line = ss.str();
    out1 << line;
    out2 << line;
}

// ── Entry point ──────────────────────────────────────────────────────────────

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

    // ── Open data file for simultaneous write ─────────────────────────
    std::ofstream data_file("sim_data.jsonl");
    if (!data_file.is_open())
    {
        std::cerr
            << "[sim] Warning: could not open sim_data.jsonl for writing\n";
    }

    // ── Emit params header (first line) ───────────────────────────────
    // The visualiser reads this to configure axis limits and limit lines.
    auto emitLine = [&](const std::string& s)
    {
        std::cout << s << "\n";
        if (data_file.is_open())
        {
            data_file << s << "\n";
        }
    };

    {
        std::ostringstream hdr;
        hdr << std::fixed << std::setprecision(6);
        hdr << "{\"params\":{"
            << "\"N\":" << p.N << ","
            << "\"dt\":" << p.dt << ","
            << "\"v_ref\":" << p.v_ref << ","
            << "\"v_max\":" << p.v_max << ","
            << "\"omega_max\":" << p.omega_max << ","
            << "\"d_hard\":" << p.d_hard << ","
            << "\"steps\":" << 300 << "}}";
        emitLine(hdr.str());
    }

    // ── Simulation loop ───────────────────────────────────────────────
    for (int t = 0; t < 300; ++t)
    {
        // At t=150 simulate a path update to test reference blending.
        if (t == 150)
        {
            path = makeStraightPath(5.0, 15.0, 0.3);
            std::cerr << "[sim] path updated at step " << t << "\n";
        }

        const Control u = ctrl.update(x, path);
        const DebugInfo& dbg = ctrl.debugInfo();

        // Write frame BEFORE stepping the plant so the logged state
        // matches the control that was just computed.
        writeFrame(std::cout, data_file, t, x, u, path, dbg);
        std::cout.flush();

        x = plantStep(x, u, p.dt);
    }

    if (data_file.is_open())
    {
        std::cerr << "[sim] data written to sim_data.jsonl\n";
    }

    return 0;
}
