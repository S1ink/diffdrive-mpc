#pragma once
// =============================================================================
// sim_utils.hpp - Simulation testing utilities for sim_main.cpp
//
// Bundles together all test-only concerns so that sim_main.cpp can focus
// on scenario selection and the top-level simulation loop:
//
//   - plantStep()       - nonlinear ground-truth plant model
//   - Path factories    - makeStraightPath, makeSharpLPath, makeFigure8Path,
//                         loadPathFromFile
//   - StateNoise        - configurable Gaussian position/heading noise
//   - StuckDetector     - sliding-window net-displacement + speed check
//   - FrameLogger       - JSON-lines frame builder and file writer
// =============================================================================

#include "mpc/path.hpp"
#include "mpc/types.hpp"
#include "mpc/params.hpp"
#include "mpc/mpc_controller.hpp"

#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <utility>


namespace mpc
{
namespace sim
{

// Load MPCParams from a plain-text key = value file.
// Throws std::runtime_error if the file cannot be opened.
MPCParams loadParamsFromFile(const std::string& filepath);


// Plant model

// Nonlinear differential-drive ground-truth plant.
// Integrates (x, y, theta) forward by one timestep using the exact kinematics.
State plantStep(const State& x, const Control& u, double dt);


// Path factories

// Horizontal straight path from x_start to x_end at constant y.
Path makeStraightPath(
    double x_start,
    double x_end,
    double y = 0.0,
    int pts = 20);

// Sharp 90-degree L-shaped path (tests corner deceleration).
Path makeSharpLPath();

// Lemniscate (figure-8) path parameterised by arc count and scale.
Path makeFigure8Path(int n = 100, double scale = 3.0);

// Load a whitespace-delimited "x y" text file as a Path.
Path loadPathFromFile(const std::string& filepath);

// Generate a random driveable path starting from (start_x, start_y) in
// direction start_theta.
//
// @param complexity  1-5.  Controls the number of segments, their length, and
//                    the maximum per-segment heading change:
//                      1 - 4 gentle segments, small turns
//                      5 - 12 longer segments, sharper turns
// @param rng         Caller-owned RNG so the sequence is reproducible.
//
// The path is densely sampled (20 pts / segment) and is guaranteed to be
// forward-only (no reversals), making it always trackable by the MPC.
Path makeRandomPath(
    double start_x,
    double start_y,
    double start_theta,
    int complexity,
    std::mt19937& rng);


// State noise

struct StateNoiseParams
{
    double pos_sigma = 0.01;    // std dev of position noise  [m]
    double ang_sigma = 0.0085;  // std dev of heading noise   [rad]
};

// Adds independent Gaussian noise to position (x, y) and heading (theta).
// Intended to simulate sensor/localization uncertainty in sim_main.cpp.
class StateNoise
{
public:
    explicit StateNoise(
        const StateNoiseParams& params = StateNoiseParams{},
        unsigned int seed = std::random_device{}());

    // Return a copy of x with noise applied.
    State apply(const State& x);

private:
    std::mt19937 gen_;
    std::normal_distribution<double> pos_dist_;
    std::normal_distribution<double> ang_dist_;
};

// Stuck detector

struct StuckDetectorParams
{
    int window = 100;         // sliding window length [steps]
    double dist_m = 0.05;     // net displacement threshold [m]
    double speed_mps = 0.02;  // average speed threshold [m/s]
};

// Detects when the robot has stopped making progress.
//
// isStuck() returns true when, over the last `window` steps:
//   - net displacement < dist_m, AND
//   - average path length / elapsed time < speed_mps.
class StuckDetector
{
public:
    explicit StuckDetector(
        const StuckDetectorParams& params = StuckDetectorParams{});

    // Call once per simulation step with the current robot state.
    void update(const State& x);

    // Returns true if the robot is considered stuck.
    // @param dt  Simulation timestep [s], used to compute average speed.
    bool isStuck(double dt) const;

private:
    StuckDetectorParams params_;
    std::vector<std::pair<double, double>> history_;  // (x, y) ring buffer
};


// Frame logger

// Writes simulation output as JSON-lines to stdout and optionally to a file.
//
// Two record types are emitted:
//   - A header record (once, via logParams) with MPC params for the visualiser.
//   - Per-step frame records (via logFrame) with robot state, control,
//     debug info, and the current path geometry.
//
// Both stdout and the file receive every record so that the visualiser can
// read directly from the binary's stdout or replay from the saved file.
class FrameLogger
{
public:
    // @param filepath  Path for the JSONL output file.  Pass an empty string
    //                  to suppress file output (stdout-only).
    explicit FrameLogger(const std::string& filepath = "sim_data.jsonl");
    ~FrameLogger();

    // Emit the params header record.  Call once before the sim loop.
    void logParams(const MPCParams& p, int max_steps);

    // Emit one frame record for the current simulation step.
    void logFrame(
        double sim_t,
        const State& x,
        const Control& u,
        const Path& path,
        const DebugInfo& dbg);

private:
    std::ofstream file_;

    // Internal helpers
    static std::string buildFrame(
        double sim_t,
        const State& x,
        const Control& u,
        const Path& path,
        const DebugInfo& dbg);

    static void jd(std::ostream& s, double v, int prec = 5);
};

}  // namespace sim
}  // namespace mpc
