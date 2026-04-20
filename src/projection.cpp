#include "mpc/projection.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace mpc
{

ProjectionResult Projector::project(const State& x, const Path& path)
{
    if (!path.valid())
    {
        throw std::invalid_argument("Path must have >= 2 points");
    }

    const size_t n_seg = path.size() - 1;
    const Eigen::Vector2d p(x.x, x.y);

    // Clamp last_segment_ to a valid range after path changes.
    if (last_segment_ >= n_seg)
    {
        last_segment_ = n_seg - 1;
    }

    // ── Forward-only bounded search ───────────────────────────────────
    // Start from last_segment_ and search at most `look_ahead` segments
    // ahead.  We never search behind last_segment_, which keeps the robot
    // from snapping back to an already-passed segment.
    const size_t start = last_segment_;
    const size_t end = std::min(n_seg, last_segment_ + look_ahead);

    ProjectionResult best;
    best.segment_index = start;
    double best_dist = std::numeric_limits<double>::max();

    for (size_t i = start; i < end; ++i)
    {
        const Eigen::Vector2d A = path.pts[i].pos;
        const Eigen::Vector2d B = path.pts[i + 1].pos;
        const Eigen::Vector2d AB = B - A;
        const double len_sq = AB.squaredNorm();

        // Parametric projection, clamped to [0,1]
        double t = 0.0;
        if (len_sq > 1e-12)
        {
            t = std::clamp((p - A).dot(AB) / len_sq, 0.0, 1.0);
        }

        const Eigen::Vector2d proj = A + t * AB;
        const double d = (p - proj).norm();

        if (d < best_dist)
        {
            best_dist = d;
            best.segment_index = i;
            best.t = t;
            best.proj = proj;
        }
    }

    // ── Hysteresis: advance last_segment_ conservatively ─────────────
    // If the best match is ahead of where we were, always accept it.
    // If the best match is on last_segment_, only advance when t > threshold
    // so we don't flip to the next segment prematurely.
    if (best.segment_index > last_segment_)
    {
        last_segment_ = best.segment_index;
    }
    else if (
        best.segment_index == last_segment_ && best.t > seg_advance_t &&
        last_segment_ + 1 < n_seg)
    {
        last_segment_++;
    }

    return best;
}

}  // namespace mpc
