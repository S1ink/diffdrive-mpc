#pragma once

#include "path.hpp"
#include "projection.hpp"
#include "params.hpp"

#include <vector>

namespace mpc
{

/// Output of ReferenceGenerator: the velocity profile only.
/// All geometric reference state / corridor normal computation has been
/// removed — Frenet-frame MPC derives those quantities directly from the
/// arc-length state s_k.
struct Reference
{
    /// Desired forward speed at each horizon step k = 0 … N [m/s].
    /// Computed by the look-ahead braking integrator; respects a_max and
    /// corner turn-rate limits.
    std::vector<double> v_profile;
};

class ReferenceGenerator
{
public:
    explicit ReferenceGenerator(const MPCParams& p) : params_(p) {}

    /// Build a velocity profile for one MPC horizon.
    ///
    /// @param path   Current path polyline.
    /// @param proj   Projection of the robot's latency-compensated position.
    /// @param v_cur  Current robot forward speed [m/s]; seeds the integrator.
    Reference generate(
        const Path& path,
        const ProjectionResult& proj,
        double v_cur = 0.0) const;

private:
    MPCParams params_;
    double distToEnd(const Path& path, size_t idx, double t) const;
};

}  // namespace mpc
