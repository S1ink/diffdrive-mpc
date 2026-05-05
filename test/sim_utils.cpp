// =============================================================================
// sim_utils.cpp - Simulation testing utilities (see sim_utils.hpp)
// =============================================================================

#include "sim_utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>


namespace mpc
{
namespace sim
{

MPCParams loadParamsFromFile(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error(
            "[config_loader] Cannot open config file: " + filepath);
    }

    std::unordered_map<std::string, double> kv;

    std::string line;
    int lineno = 0;
    while (std::getline(file, line))
    {
        ++lineno;

        // Strip leading whitespace
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;  // blank or comment line
        }

        const size_t eq = line.find('=', first);
        if (eq == std::string::npos)
        {
            std::cerr << "[config_loader] Line " << lineno
                      << ": missing '=', skipping.\n";
            continue;
        }

        // Extract and trim key
        std::string key = line.substr(first, eq - first);
        const size_t key_end = key.find_last_not_of(" \t");
        if (key_end == std::string::npos)
        {
            continue;
        }
        key = key.substr(0, key_end + 1);

        // Extract and trim value (strip inline comments)
        std::string val = line.substr(eq + 1);
        const size_t val_start = val.find_first_not_of(" \t");
        if (val_start == std::string::npos)
        {
            continue;  // empty value
        }
        val = val.substr(val_start);

        // Strip inline comment
        const size_t comment_pos = val.find('#');
        if (comment_pos != std::string::npos)
        {
            val = val.substr(0, comment_pos);
        }

        // Trim trailing whitespace / CR
        const size_t val_end = val.find_last_not_of(" \t\r\n");
        if (val_end == std::string::npos)
        {
            continue;
        }
        val = val.substr(0, val_end + 1);

        if (val.empty())
        {
            continue;
        }

        // Parse numeric value
        try
        {
            kv[key] = std::stod(val);
        }
        catch (const std::exception&)
        {
            std::cerr << "[config_loader] Line " << lineno
                      << ": cannot parse value for '" << key << "' = '" << val
                      << "', skipping.\n";
        }
    }

    // Known keys (used to warn about unrecognised entries)
    static const std::unordered_set<std::string> known_keys = {
        "N",
        "dt",
        "feedback_delay_s",
        "v_max",
        "v_min",
        "omega_max",
        "a_max",
        "alpha_max",
        "d_hard",
        "w_slack",
        "Q_xy",
        "Q_theta",
        "Q_xy_terminal",
        "Q_theta_terminal",
        "R_v",
        "R_omega",
        "R_rate_v",
        "R_rate_omega",
        "funnel_decay_tau",
        "blend_alpha",
        "stanley_k",
        "stanley_v_min",
        "stanley_decay",
        "goal_threshold",
        "goal_cte_scale",
        "goal_stop_vel",
        "fallback_decay",
    };
    for (const auto& [k, _] : kv)
    {
        if (!known_keys.count(k))
        {
            std::cerr << "[config_loader] Warning: unrecognised key '" << k
                      << "' will be ignored.\n";
        }
    }

    // Apply values to a default-constructed MPCParams
    MPCParams p;

    auto set = [&](const std::string& k, double& field)
    {
        const auto it = kv.find(k);
        if (it != kv.end())
        {
            field = it->second;
        }
    };
    auto setInt = [&](const std::string& k, int& field)
    {
        const auto it = kv.find(k);
        if (it != kv.end())
        {
            field = static_cast<int>(it->second);
        }
    };

    setInt("N", p.N);
    set("dt", p.dt);
    set("feedback_delay_s", p.feedback_delay_s);
    set("v_max", p.v_max);
    set("v_min", p.v_min);
    set("omega_max", p.omega_max);
    set("a_max", p.a_max);
    set("alpha_max", p.alpha_max);
    set("d_hard", p.d_hard);
    set("w_slack", p.w_slack);
    set("Q_xy", p.Q_xy);
    set("Q_theta", p.Q_theta);
    set("Q_xy_terminal", p.Q_xy_terminal);
    set("Q_theta_terminal", p.Q_theta_terminal);
    set("R_v", p.R_v);
    set("R_omega", p.R_omega);
    set("R_rate_v", p.R_rate_v);
    set("R_rate_omega", p.R_rate_omega);
    set("funnel_decay_tau", p.funnel_decay_tau);
    set("blend_alpha", p.blend_alpha);
    set("stanley_k", p.stanley_k);
    set("stanley_v_min", p.stanley_v_min);
    set("stanley_decay", p.stanley_decay);
    set("goal_threshold", p.goal_threshold);
    set("goal_cte_scale", p.goal_cte_scale);
    set("goal_stop_vel", p.goal_stop_vel);
    set("fallback_decay", p.fallback_decay);

    return p;
}

// Plant model

State plantStep(const State& x, const Control& u, double dt)
{
    return {
        x.x + u.v * std::cos(x.theta) * dt,
        x.y + u.v * std::sin(x.theta) * dt,
        x.theta + u.omega * dt};
}

// Path factories

Path makeStraightPath(double x_start, double x_end, double y, int pts)
{
    Path p;
    for (int i = 0; i <= pts; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(pts);
        p.pts.push_back({Eigen::Vector2d(x_start + t * (x_end - x_start), y)});
    }
    return p;
}

Path makeSharpLPath()
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

Path makeFigure8Path(int n, double scale)
{
    Path p;
    for (int i = 0; i <= n; ++i)
    {
        const double t = 2.0 * M_PI * i / n;
        p.pts.push_back({Eigen::Vector2d(
            scale * std::sin(t),
            scale * std::sin(t) * std::cos(t))});
    }
    return p;
}

Path loadPathFromFile(const std::string& filepath)
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

// State noise

StateNoise::StateNoise(const StateNoiseParams& params, unsigned int seed) :
    gen_(seed),
    pos_dist_(0.0, params.pos_sigma),
    ang_dist_(0.0, params.ang_sigma)
{
}

State StateNoise::apply(const State& x)
{
    return {
        x.x + pos_dist_(gen_),
        x.y + pos_dist_(gen_),
        x.theta + ang_dist_(gen_)};
}

// Stuck detector

StuckDetector::StuckDetector(const StuckDetectorParams& params) :
    params_(params)
{
    history_.reserve(params_.window + 1);
}

void StuckDetector::update(const State& x)
{
    history_.push_back({x.x, x.y});
    if (static_cast<int>(history_.size()) > params_.window)
    {
        history_.erase(history_.begin());
    }
}

bool StuckDetector::isStuck(double dt) const
{
    if (static_cast<int>(history_.size()) < params_.window)
    {
        return false;  // not enough history yet
    }

    // Net displacement over the window.
    const double dx = history_.back().first - history_.front().first;
    const double dy = history_.back().second - history_.front().second;
    const double net_disp = std::sqrt(dx * dx + dy * dy);

    // Average speed (total path length / elapsed time).
    double path_len = 0.0;
    for (int j = 1; j < params_.window; ++j)
    {
        const double ddx = history_[j].first - history_[j - 1].first;
        const double ddy = history_[j].second - history_[j - 1].second;
        path_len += std::sqrt(ddx * ddx + ddy * ddy);
    }
    const double avg_speed = path_len / (params_.window * dt);

    return (net_disp < params_.dist_m) && (avg_speed < params_.speed_mps);
}

// Frame logger

FrameLogger::FrameLogger(const std::string& filepath)
{
    if (!filepath.empty())
    {
        file_.open(filepath);
    }
}

FrameLogger::~FrameLogger() = default;

void FrameLogger::logParams(const MPCParams& p, int max_steps)
{
    std::ostringstream ss;
    ss << "{\"params\":{"
       << "\"N\":" << p.N << ",\"dt\":" << p.dt << ",\"v_max\":" << p.v_max
       << ",\"v_min\":" << p.v_min  // needed by the Python visualiser
       << ",\"omega_max\":" << p.omega_max << ",\"d_hard\":" << p.d_hard
       << ",\"max_steps\":" << max_steps << "}}";

    const std::string line = ss.str();
    std::cout << line << "\n";
    if (file_.is_open())
    {
        file_ << line << "\n";
    }
}

void FrameLogger::logFrame(
    double sim_t,
    const State& x,
    const Control& u,
    const Path& path,
    const DebugInfo& dbg)
{
    const std::string line = buildFrame(sim_t, x, u, path, dbg);
    std::cout << line << "\n";
    std::cout.flush();
    if (file_.is_open())
    {
        file_ << line << "\n";
    }
}

// FrameLogger internals

void FrameLogger::jd(std::ostream& s, double v, int prec)
{
    s << std::fixed << std::setprecision(prec) << v;
}

std::string FrameLogger::buildFrame(
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

    ss << "\"robot\":{"
       << "\"x\":" << x.x << ",\"y\":" << x.y << ",\"theta\":" << x.theta
       << "},";

    ss << "\"control\":{"
       << "\"v\":" << u.v << ",\"omega\":" << u.omega << "},";

    ss << "\"cte\":";
    jd(ss, dbg.cte_raw);
    ss << ",";

    ss << "\"proj\":[";
    jd(ss, dbg.proj_pt.x());
    ss << ",";
    jd(ss, dbg.proj_pt.y());
    ss << "],";

    ss << "\"solver_ok\":" << (dbg.solver_ok ? "true" : "false") << ",";

    ss << "\"solve_ms\":";
    jd(ss, dbg.solve_ms, 4);
    ss << ",";

    ss << "\"pred\":[";
    for (int k = 0; k < static_cast<int>(dbg.pred_traj.size()); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[" << dbg.pred_traj[k].x << "," << dbg.pred_traj[k].y << ","
           << dbg.pred_traj[k].theta << "]";
    }
    ss << "],";

    ss << "\"ref\":[";
    for (int k = 0; k < static_cast<int>(dbg.ref_snap.x_ref.size()); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[" << dbg.ref_snap.x_ref[k].x << "," << dbg.ref_snap.x_ref[k].y
           << "," << dbg.ref_snap.x_ref[k].theta << "]";
    }
    ss << "],";

    ss << "\"path\":[";
    for (int i = 0; i < static_cast<int>(path.size()); ++i)
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

}  // namespace sim
}  // namespace mpc
