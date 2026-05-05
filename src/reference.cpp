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
// Fix (first pass):
//   A PathSmoother is built at the start of each generate() call.  It
//   replaces each sharp corner with a circular arc whose radius is
//   bounded by v_max/ω_max and fitted within the adjacent segment lengths
//   via a multi-pass junction optimizer.  All reference positions, headings,
//   corridor normals, and speed limits are then sampled from this smooth
//   geometry instead of the raw polyline.
//
// Fix (second pass — this file):
//   The arc-position seed s0 is now obtained by projecting the robot's actual
//   2-D position directly onto the smooth path, rather than by projecting
//   proj.proj (the closest point on the raw polyline) and then re-projecting
//   that raw point onto the smooth path.
//
//   The old two-step approach caused a discrete jump in s0 whenever the
//   stateful raw Projector crossed the bisector plane at a corner and advanced
//   last_segment_: proj.proj would snap from a point near B on segment A→B to
//   a point near B on segment B→C, moving s0 by as much as half the arc
//   length in a single cycle (~½·r·θ, which can be ~0.8 m for typical corners).
//   The horizon keypoints — which start from s0 — then shifted abruptly,
//   producing a large discontinuity in the reference trajectory.
//
//   Projecting robot_pos directly onto the smooth path removes this artefact:
//   the smooth path is C1 (position and tangent are continuous through arcs),
//   so the nearest point on it to the robot changes continuously as the robot
//   traverses the corner.
//
// What is preserved:
//   • The forward velocity integration with look-ahead braking (§4 below) is
//     unchanged in structure.  Velocity events at arc entries replace the
//     old heuristic ω_max·ds/Δθ formula and are exact by construction.
//   • The seg_normals / proj_pts separation (physical corridor vs. lookahead
//     reference) is preserved: expected_s tracks where the robot is expected
//     to be on the smooth path, independently of the lookahead arc position.
//   • The public API (signature of generate()) is unchanged except for the
//     addition of the robot_pos parameter.
// =============================================================================

#include "mpc/reference.hpp"

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

// ── Main generator ────────────────────────────────────────────────────────────

Reference ReferenceGenerator::generate(
    const Path& path,
    const ProjectionResult& proj,
    const Eigen::Vector2d& robot_pos,
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
    const PathSmoother::SmoothedPath sp = smoother_.smooth(path);

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
    // Project the robot's actual 2-D position directly onto the smooth path
    // to obtain the arc-position seed s0.
    //
    // The previous approach projected proj.proj (the closest point on the
    // RAW polyline) onto the smooth path.  Near corners, that caused a
    // discrete jump: when the raw Projector crossed the bisector plane and
    // snapped last_segment_ from i to i+1, proj.proj moved discontinuously,
    // pulling s0 forward by up to half an arc length in a single cycle.
    //
    // Using robot_pos instead eliminates the jump because:
    //   • The smooth path is C1 (position and tangent are continuous through
    //     the corner arc), so the nearest point on it to the robot changes
    //     continuously as the robot traverses the corner.
    //   • robot_pos is the latency-compensated state and changes by at most
    //     v_max·dt ≈ 0.06 m per cycle, producing a proportionally small
    //     change in s0.
    //
    // Note on backward-snapping: SmoothedPath::project() is a global
    // nearest-neighbour search.  For paths without hairpin segments whose
    // legs are closer together than the robot-to-path distance, the global
    // minimum always lands on the correct forward position.  If a pathological
    // hairpin is possible in your application, the caller can add a monotone
    // floor by passing the previous s0 as an additional argument.
    const auto [s0, smooth_proj] = sp.project(robot_pos);

    // CTE measured against the smooth arc, not the raw polyline.
    // At corners the smooth projection stays on the arc, so this avoids the
    // inflated CTE that the raw-polyline projector produces during a turn.
    {
        const PathSmoother::SmoothSample s = sp.sampleAt(s0, params_.v_max);
        r.cte = s.normal.dot(robot_pos - smooth_proj);
    }

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
