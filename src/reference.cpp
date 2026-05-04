// =============================================================================
// reference.cpp — MPC Reference Generator (smooth-path edition)
//
// Key change from the original:
//   The original generator sampled reference positions and headings directly
//   from the raw polyline.  At corners, this caused the heading to jump
//   instantaneously from the incoming segment direction to the outgoing one
//   — a step that is kinematically infeasible for a diff-drive robot with
//   bounded angular velocity ω_max.  The QP was then handed targets it could
//   not reach within the horizon, causing the robot to stall.
//
// Fix:
//   A PathSmoother is built at the start of each generate() call.  It
//   replaces each sharp corner with a circular arc whose radius is
//   bounded by v_max/ω_max and fitted within the adjacent segment lengths
//   via a multi-pass junction optimizer.  All reference positions, headings,
//   corridor normals, and speed limits are then sampled from this smooth
//   geometry instead of the raw polyline.
//
// What is preserved:
//   • The forward velocity integration with look-ahead braking (§4 below) is
//     unchanged in structure.  Velocity events at arc entries replace the
//     old heuristic ω_max·ds/Δθ formula and are exact by construction.
//   • The seg_normals / proj_pts separation (physical corridor vs. lookahead
//     reference) is preserved: expected_s tracks where the robot is expected
//     to be on the smooth path, independently of the lookahead arc position.
//   • The public API (signature of generate()) is unchanged.
// =============================================================================

#include "mpc/reference.hpp"
#include "mpc/path_smoother.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mpc
{

// ── Internal velocity-event type ─────────────────────────────────────────────

struct VelocityEvent
{
    double s;      // arc-length position from smooth path start [m]
    double v_lim;  // maximum speed when arriving at s [m/s]
};

// ── distToEnd ─────────────────────────────────────────────────────────────────

double ReferenceGenerator::distToEnd(const Path& path, size_t idx, double t)
    const
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

// ── Main generator ────────────────────────────────────────────────────────────

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj,
    double v_cur) const
{
    const int N = params_.N;
    const double dt = params_.dt;

    Reference r;
    r.x_ref.resize(N + 1);
    r.seg_normals.resize(N + 1);
    r.proj_pts.resize(N + 1);
    r.v_profile.resize(N + 1);

    // ── 1. Build smooth geometry ───────────────────────────────────────
    //
    // PathSmoother replaces each interior waypoint with a circular arc.
    // All subsequent sampling is done on this smooth path rather than on
    // the raw polyline, giving kinematically continuous reference headings.
    PathSmoother smoother(params_);
    const PathSmoother::SmoothedPath sp = smoother.smooth(path);

    // ── Fallback: degenerate / very short path ─────────────────────────
    if (sp.empty() || sp.total < 1e-6)
    {
        const Eigen::Vector2d fallback_pos =
            path.pts.empty() ? Eigen::Vector2d::Zero() : path.pts.back().pos;
        const Eigen::Vector2d fallback_n{0.0, 1.0};
        for (int k = 0; k <= N; ++k)
        {
            r.x_ref[k] = {fallback_pos.x(), fallback_pos.y(), 0.0};
            r.seg_normals[k] = fallback_n;
            r.proj_pts[k] = fallback_pos;
            r.v_profile[k] = 0.0;
        }
        return r;
    }

    // ── 2. Arc position of the current robot projection ────────────────
    //
    // proj.proj is the closest point on the RAW polyline to the robot.
    // Projecting it onto the smooth path gives the corresponding arc
    // position s0.  Since the smooth path deviates from the raw polyline
    // only near corners (by at most one arc radius), this is accurate
    // everywhere and exact between corners.
    const auto [s0, _smooth_proj] = sp.project(proj.proj);

    // ── Edge case: robot already at or past path end ───────────────────
    if (s0 >= sp.total - 1e-6)
    {
        const PathSmoother::SmoothSample end =
            sp.sampleAt(sp.total, params_.v_max);
        for (int k = 0; k <= N; ++k)
        {
            r.x_ref[k] = {end.pos.x(), end.pos.y(), end.heading};
            r.seg_normals[k] = end.normal;
            r.proj_pts[k] = end.pos;
            r.v_profile[k] = 0.0;
        }
        return r;
    }

    // ── 3. Build velocity events from smooth path ─────────────────────
    //
    // Each ArcSegment contributes a speed cap event at its entry point:
    //   v_cap = arc.v_max  (= radius · ω_max, exact kinematic limit)
    // The backward-pass integrator in §4 ensures the robot brakes in time.
    //
    // A mandatory zero-speed event at path end provides the stopping goal.
    std::vector<VelocityEvent> events;
    events.push_back({sp.total, 0.0});  // always stop at path end

    for (int i = 0; i < static_cast<int>(sp.segs.size()); ++i)
    {
        if (!std::holds_alternative<PathSmoother::ArcSegment>(sp.segs[i]))
        {
            continue;
        }

        const auto& arc = std::get<PathSmoother::ArcSegment>(sp.segs[i]);
        const double s_arc = sp.cum[i];  // arc entry on smooth path

        // Only include arcs that lie ahead of the current position
        if (s_arc > s0 + 1e-6)
        {
            events.push_back({s_arc, arc.v_max});
        }
    }

    // ── 4. Forward velocity integration with look-ahead braking ───────
    //
    // At each horizon step k the speed is capped at the tightest value
    // from which the robot can brake to every upcoming event within that
    // event's distance:
    //
    //   v_cap_ev = sqrt( v_ev² + 2·a_max·(s_ev − s_current) )
    //
    // This is identical in structure to the original generator; only the
    // source of the events differs (smooth arc entries vs. raw waypoints).
    std::vector<double> arc_at(N + 1, 0.0);
    {
        double v = std::clamp(v_cur, 0.0, params_.v_max);

        for (int k = 0; k < N; ++k)
        {
            const double s_cur = s0 + arc_at[k];

            // Look-ahead: tightest braking cap over all future events
            double v_cap = params_.v_max;
            for (const auto& ev : events)
            {
                const double ds = ev.s - s_cur;
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

            // Also cap to the smooth path's instantaneous speed limit at the
            // current arc position (catches the case where the robot has
            // entered an arc before the integration started braking for it)
            const auto samp_now = sp.sampleAt(s_cur, params_.v_max);
            v_cap = std::min(v_cap, samp_now.v_limit);

            // Advance velocity within ±a_max·dt
            v = std::clamp(
                v_cap,
                v - params_.a_max * dt,
                v + params_.a_max * dt);
            v = std::max(v, 0.0);

            r.v_profile[k] = v;
            arc_at[k + 1] = arc_at[k] + v * dt;
        }

        // Terminal slot: repeat last velocity (consumed by the lineariser)
        r.v_profile[N] = (N > 0) ? r.v_profile[N - 1] : 0.0;
    }

    // ── 5. Sample reference from smooth path ──────────────────────────
    //
    // Two arc positions are tracked independently:
    //
    //   s_ref  — the LOOKAHEAD reference position (s0 + arc_at[k]).
    //            Used for x_ref[k].  It may be ahead of the robot,
    //            giving the optimizer a "carrot" to chase.
    //
    //   expected_s — the PHYSICAL corridor position (where the robot is
    //                expected to be at step k, advancing by v[k]·dt).
    //                Used for seg_normals[k] and proj_pts[k].
    //                This separates corridor geometry from the lookahead
    //                reference, which is essential for the QP half-space
    //                constraints to be correct.
    //
    // Both are sampled from the smooth path via sampleAt(), which returns
    // the correct tangent heading (and arc-normal) regardless of whether
    // the position is on a line or arc segment.

    double expected_s = s0;

    for (int k = 0; k <= N; ++k)
    {
        // ── Reference (lookahead) ──────────────────────────────────────
        const double s_ref = std::min(s0 + arc_at[k], sp.total);
        const auto ref_samp = sp.sampleAt(s_ref, params_.v_max);

        r.x_ref[k] = {ref_samp.pos.x(), ref_samp.pos.y(), ref_samp.heading};

        // ── Physical corridor ──────────────────────────────────────────
        const auto phys_samp =
            sp.sampleAt(std::min(expected_s, sp.total), params_.v_max);

        r.seg_normals[k] = phys_samp.normal;
        r.proj_pts[k] = phys_samp.pos;

        // Advance expected robot arc position for the next step
        if (k < N)
        {
            expected_s += r.v_profile[k] * dt;
        }
    }

    return r;
}

}  // namespace mpc
