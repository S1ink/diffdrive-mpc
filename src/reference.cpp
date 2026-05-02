#include "mpc/reference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mpc
{

struct VelocityEvent
{
    double s;      // arc-length position from path start [m]
    double v_lim;  // maximum speed at this event [m/s]
};

static std::vector<VelocityEvent> buildConstraintEvents(
    const Path& path,
    const std::vector<double>& cum,
    double s0,
    double v_max,
    double omega_max)
{
    std::vector<VelocityEvent> events;
    const int n = (int)path.size();

    events.push_back({cum.back(), 0.0});  // stop at end

    for (int i = 1; i < n - 1; ++i)
    {
        const double s_j = cum[i];
        if (s_j <= s0 + 1e-6)
        {
            continue;
        }

        const Eigen::Vector2d d_in =
            (path.pts[i].pos - path.pts[i - 1].pos).normalized();
        const Eigen::Vector2d d_out =
            (path.pts[i + 1].pos - path.pts[i].pos).normalized();

        const double cos_a = std::clamp(d_in.dot(d_out), -1.0, 1.0);
        const double angle = std::acos(cos_a);

        if (angle < 0.05)
        {
            continue;
        }

        const double l_in = (path.pts[i].pos - path.pts[i - 1].pos).norm();
        const double l_out = (path.pts[i + 1].pos - path.pts[i].pos).norm();
        const double ds_avg = 0.5 * (l_in + l_out);

        const double v_lim =
            std::clamp(omega_max * ds_avg / (angle + 1e-9), 0.0, v_max);
        events.push_back({s_j, v_lim});
    }

    return events;
}

double ReferenceGenerator::distToEnd(
    const Path& path,
    size_t idx,
    double t) const
{
    if (idx >= path.size() - 1)
    {
        return 0.0;
    }
    const double seg_len = (path.pts[idx + 1].pos - path.pts[idx].pos).norm();
    double d = (1.0 - t) * seg_len;
    for (size_t i = idx + 1; i < path.size() - 1; ++i)
    {
        d += (path.pts[i + 1].pos - path.pts[i].pos).norm();
    }
    return d;
}

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj,
    double v_cur) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    Reference r;
    r.v_profile.resize(N + 1, 0.0);

    const int n_pts = (int)path.size();

    // ── 1. Cumulative arc lengths ──────────────────────────────────────
    const std::vector<double> cum = cumulativeArcs(path);
    const double total = cum.back();

    std::vector<double> segl(n_pts - 1);
    for (int i = 0; i < n_pts - 1; ++i)
    {
        segl[i] = cum[i + 1] - cum[i];
    }

    // ── 2. Arc position of the current projection ──────────────────────
    const size_t seg0 = proj.segment_index;
    const double s0 = cum[seg0] + proj.t * segl[seg0];

    if (s0 >= total - 1e-6)
    {
        return r;  // already at end; v_profile stays zero
    }

    // ── 3. Build constraint events ─────────────────────────────────────
    const std::vector<VelocityEvent> events =
        buildConstraintEvents(path, cum, s0, params_.v_max, params_.omega_max);

    // ── 4. Forward velocity integration with look-ahead braking ───────
    std::vector<double> arc_at(N + 1, 0.0);
    {
        double v = std::clamp(v_cur, 0.0, params_.v_max);

        for (int k = 0; k < N; ++k)
        {
            double v_cap = params_.v_max;
            for (const auto& ev : events)
            {
                const double ds = (ev.s - s0) - arc_at[k];
                if (ds >= 0.0)
                {
                    v_cap = std::min(
                        v_cap,
                        std::sqrt(
                            std::max(
                                0.0,
                                ev.v_lim * ev.v_lim +
                                    2.0 * params_.a_max * ds)));
                }
            }

            v = std::clamp(
                v_cap,
                v - params_.a_max * dt,
                v + params_.a_max * dt);
            v = std::max(v, 0.0);

            r.v_profile[k] = v;
            arc_at[k + 1] = arc_at[k] + v * dt;
        }

        r.v_profile[N] = (N > 0) ? r.v_profile[N - 1] : 0.0;
    }

    return r;
}

}  // namespace mpc
