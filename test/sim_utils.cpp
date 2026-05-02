// =============================================================================
// sim_utils.cpp  —  Simulation testing utilities (see sim_utils.hpp)
// =============================================================================

#include "sim_utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace mpc
{
namespace sim
{

// ── Plant model ───────────────────────────────────────────────────────────────

State plantStep(const State& x, const Control& u, double dt)
{
    return {
        x.x + u.v * std::cos(x.theta) * dt,
        x.y + u.v * std::sin(x.theta) * dt,
        x.theta + u.omega * dt};
}

// ── Path factories ────────────────────────────────────────────────────────────

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

// ── StateNoise ────────────────────────────────────────────────────────────────

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

// ── StuckDetector ─────────────────────────────────────────────────────────────

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

// ── FrameLogger ───────────────────────────────────────────────────────────────

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
       << "\"N\":" << p.N << ",\"dt\":" << p.dt //<< ",\"v_ref\":" << p.v_ref
       << ",\"v_max\":" << p.v_max << ",\"omega_max\":" << p.omega_max
       << ",\"d_hard\":" << p.d_hard << ",\"max_steps\":" << max_steps << "}}";

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

// ── FrameLogger internals ─────────────────────────────────────────────────────

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

    ss << "\"d_hard_eff\":";
    jd(ss, dbg.d_hard_eff);
    ss << ",";

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
    for (int k = 0; k < static_cast<int>(dbg.ref_traj.size()); ++k)
    {
        if (k)
        {
            ss << ",";
        }
        ss << "[" << dbg.ref_traj[k].x << "," << dbg.ref_traj[k].y << ","
           << dbg.ref_traj[k].theta << "]";
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
