#!/usr/bin/env python3
"""
plot_sim.py  —  MPC Path Follower Visualizer
=============================================

Runs the C++ sim binary (or reads a saved .jsonl file), then plays back
the results as a 20 Hz animation.  All five panels update together:

    ┌─────────────────────────┬──────────────┐
    │                         │   v(t)       │
    │   Spatial XY            ├──────────────┤
    │   (path, corridor,      │   ω(t)       │
    │    robot, horizons)     │              │
    ├────────────────────────────────────────┤
    │         Cross-track error (full width) │
    └────────────────────────────────────────┘

Usage
-----
    # Run binary, animate, also save data to sim_data.jsonl:
    python tools/plot_sim.py --binary ./build/sim_test

    # Replay a previously saved file:
    python tools/plot_sim.py --file sim_data.jsonl

    # Run binary and save animation to MP4:
    python tools/plot_sim.py --binary ./build/sim_test --save sim.mp4

    # Run binary, save data to custom path, save animation as GIF:
    python tools/plot_sim.py --binary ./build/sim_test \\
                             --data  my_run.jsonl \\
                             --save  my_run.gif
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (dark theme)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg_fig":        "#12121f",
    "bg_ax":         "#0d0d1a",
    "spine":         "#2a2a44",
    "tick":          "#9090bb",
    "title":         "#d0d0ff",
    "label":         "#7777aa",
    "path":          "#44445a",
    "corridor_face": "#2244aa",
    "corridor_edge": "#4466cc",
    "robot_trace":   "#4488ff",
    "ref_horizon":   "#44dd66",
    "pred_horizon":  "#ff6633",
    "proj_pt":       "#ffcc00",
    "robot_dot":     "#ffffff",
    "v_line":        "#44aaff",
    "v_ref":         "#44ff88",
    "v_max":         "#ff4444",
    "w_line":        "#ffaa33",
    "w_max":         "#ff4444",
    "cte_line":      "#cc44ff",
    "cte_band":      "#ff4444",
    "solver_fail":   "#ff2222",
    "path_update":   "#ffff00",
    "cursor":        "#ffffff",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_binary(binary_path: str, data_path: str | None = None):
    print(f"[plot_sim] Running {binary_path} …", file=sys.stderr)
    result = subprocess.run(
        [binary_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[plot_sim] Binary exited with code {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    lines = result.stdout.splitlines()

    if data_path:
        with open(data_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[plot_sim] Data saved → {data_path}", file=sys.stderr)

    return _parse_lines(lines)


def load_from_file(filepath: str):
    with open(filepath) as f:
        lines = f.read().splitlines()
    return _parse_lines(lines)


def _parse_lines(lines):
    params = {}
    frames = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "params" in obj:
            params = obj["params"]
        else:
            frames.append(obj)
    return params, frames


# ─────────────────────────────────────────────────────────────────────────────
# Geometry & Bounds
# ─────────────────────────────────────────────────────────────────────────────

def _offset_polyline(pts: np.ndarray, d: float):
    n = len(pts)
    normals = np.empty_like(pts)
    for i in range(n):
        if i == 0:
            seg = pts[1] - pts[0]
        elif i == n - 1:
            seg = pts[-1] - pts[-2]
        else:
            seg = pts[i + 1] - pts[i - 1]
        length = np.linalg.norm(seg)
        if length < 1e-9:
            normals[i] = [0.0, 1.0]
        else:
            normals[i] = [-seg[1] / length, seg[0] / length]
    return pts + d * normals, pts - d * normals


def corridor_polygon(pts: np.ndarray, d: float) -> np.ndarray:
    left, right = _offset_polyline(pts, d)
    return np.vstack([left, right[::-1]])


def compute_xy_bounds(frames, pad=0.8, min_span=4.0):
    xs, ys = [], []
    for f in frames:
        xs.append(f["robot"]["x"])
        ys.append(f["robot"]["y"])
        for pt in f.get("path", []):
            xs.append(pt[0])
            ys.append(pt[1])
        # Include ref and pred as well
        for pt in f.get("ref", []):
            xs.append(pt[0])
            ys.append(pt[1])

    if not xs: 
        return (-min_span/2, min_span/2, -min_span/2, min_span/2)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Calculate current spans
    x_span = x_max - x_min
    y_span = y_max - y_min

    # If the span is too small (e.g., a straight line), 
    # expand from the center to at least min_span
    if x_span < min_span:
        center_x = (x_min + x_max) / 2
        x_min = center_x - min_span / 2
        x_max = center_x + min_span / 2

    if y_span < min_span:
        center_y = (y_min + y_max) / 2
        y_min = center_y - min_span / 2
        y_max = center_y + min_span / 2

    return (x_min - pad, x_max + pad, 
            y_min - pad, y_max + pad)


def compute_time_bounds(frames, params):
    """Step 1: Precompute global bounds for all time-series axes."""
    t_all   = [f["t"] for f in frames]
    v_all   = [f["control"]["v"] for f in frames]
    w_all   = [f["control"]["omega"] for f in frames]
    cte_all = [f["cte"] for f in frames]

    t_min, t_max = min(t_all), max(t_all)

    v_max_param = params.get("v_max", 0.5)
    omega_max   = params.get("omega_max", 1.2)
    d_hard      = params.get("d_hard", 0.1)

    v_lim = max(max(v_all), v_max_param)
    w_lim = max(max(abs(x) for x in w_all), omega_max)
    cte_lim = max(max(abs(x) for x in cte_all), d_hard)

    return {
        "t":   (t_min, t_max + 2),
        "v":   (-0.05, v_lim * 1.2),
        "w":   (-w_lim * 1.25, w_lim * 1.25),
        "cte": (-cte_lim * 1.35, cte_lim * 1.35),
    }


def find_path_updates(frames):
    events = []
    prev_path = None
    for f in frames:
        cur_path = [tuple(pt) for pt in f["path"]]
        if prev_path is not None and cur_path != prev_path:
            events.append(f["t"])
        prev_path = cur_path
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Figure construction
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(C["bg_ax"])
    ax.tick_params(colors=C["tick"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["spine"])


def build_figure():
    fig = plt.figure(figsize=(15, 8), facecolor=C["bg_fig"])
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.75],
                           hspace=0.50, wspace=0.32, left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    ax_xy  = fig.add_subplot(gs[0:2, 0])
    ax_v   = fig.add_subplot(gs[0, 1])
    ax_w   = fig.add_subplot(gs[1, 1])
    ax_cte = fig.add_subplot(gs[2, :])

    for ax in (ax_xy, ax_v, ax_w, ax_cte):
        _style_ax(ax)

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("Spatial (XY)", color=C["title"], fontsize=10, pad=6)
    ax_v.set_title("Forward velocity", color=C["title"], fontsize=9, pad=4)
    ax_w.set_title("Angular velocity", color=C["title"], fontsize=9, pad=4)
    ax_cte.set_title("Cross-track error", color=C["title"], fontsize=9, pad=4)

    return fig, ax_xy, ax_v, ax_w, ax_cte


def add_static_lines(ax_v, ax_w, ax_cte, params):
    kw = dict(linewidth=0.9, alpha=0.65, linestyle="--")
    v_ref, v_max = params.get("v_ref", 0.3), params.get("v_max", 0.5)
    omega_max, d_hard = params.get("omega_max", 1.2), params.get("d_hard", 0.1)

    ax_v.axhline(v_ref, color=C["v_ref"], label=f"v_ref={v_ref:.2f}", **kw)
    ax_v.axhline(v_max, color=C["v_max"], label=f"v_max={v_max:.2f}", **kw)
    ax_w.axhline(omega_max, color=C["w_max"], label=f"±ω={omega_max:.2f}", **kw)
    ax_w.axhline(-omega_max, color=C["w_max"], **kw)
    ax_cte.axhline(d_hard, color=C["cte_band"], label=f"±d={d_hard:.3f}", **kw)
    ax_cte.axhline(-d_hard, color=C["cte_band"], **kw)

    for ax in (ax_v, ax_w, ax_cte):
        ax.legend(loc="upper right", fontsize=7, facecolor=C["bg_ax"], 
                  edgecolor=C["spine"], labelcolor=C["tick"])


def init_artists(ax_xy, ax_v, ax_w, ax_cte):
    arts = {}
    # XY Artists
    arts["corridor"] = MplPolygon(np.zeros((4, 2)), closed=True, facecolor=C["corridor_face"], 
                                  alpha=0.20, edgecolor=C["corridor_edge"], linewidth=0.8, zorder=1)
    ax_xy.add_patch(arts["corridor"])
    arts["path_line"], = ax_xy.plot([], [], color=C["path"], linewidth=2.0, zorder=2)
    arts["robot_trace"], = ax_xy.plot([], [], color=C["robot_trace"], linewidth=1.2, alpha=0.65, zorder=3)
    arts["ref_h"], = ax_xy.plot([], [], "o--", color=C["ref_horizon"], markersize=3, alpha=0.8)
    arts["pred_h"], = ax_xy.plot([], [], "o-", color=C["pred_horizon"], markersize=3, alpha=0.85)
    arts["proj"], = ax_xy.plot([], [], "x", color=C["proj_pt"], markersize=8, markeredgewidth=2.0)
    arts["robot_dot"], = ax_xy.plot([], [], "o", color=C["robot_dot"], markersize=7, zorder=8)
    arts["heading"] = ax_xy.quiver(0, 0, 0, 0, color=C["robot_dot"], scale=5.0, scale_units="inches", width=0.005)
    arts["step_text"] = ax_xy.text(0.98, 0.97, "", transform=ax_xy.transAxes, ha="right", va="top", color=C["tick"], fontsize=8)

    # Time-series Artists
    arts["v_line"], = ax_v.plot([], [], color=C["v_line"], linewidth=1.2)
    arts["w_line"], = ax_w.plot([], [], color=C["w_line"], linewidth=1.2)
    arts["cte_line"], = ax_cte.plot([], [], color=C["cte_line"], linewidth=1.2)

    # Step 5: Add vertical cursor lines
    cursor_kw = dict(color=C["cursor"], alpha=0.3, linewidth=1, linestyle=":")
    arts["cursor_v"] = ax_v.axvline(0, **cursor_kw)
    arts["cursor_w"] = ax_w.axvline(0, **cursor_kw)
    arts["cursor_cte"] = ax_cte.axvline(0, **cursor_kw)

    return arts


# ─────────────────────────────────────────────────────────────────────────────
# Animation update closure
# ─────────────────────────────────────────────────────────────────────────────

def make_update(frames, arts, ax_v, ax_w, ax_cte, params, path_events):
    # Step 4: Preallocate arrays for smoother rendering
    num_f = len(frames)
    robot_x = np.zeros(num_f)
    robot_y = np.zeros(num_f)
    t_hist  = np.zeros(num_f)
    v_hist  = np.zeros(num_f)
    w_hist  = np.zeros(num_f)
    cte_hist = np.zeros(num_f)

    d_hard = params.get("d_hard", 0.10)
    drawn_events = set()

    def update(fi):
        nonlocal drawn_events
        f = frames[fi]
        t, rx, ry = f["t"], f["robot"]["x"], f["robot"]["y"]
        rth, v, w, cte = f["robot"]["theta"], f["control"]["v"], f["control"]["omega"], f["cte"]
        
        robot_x[fi], robot_y[fi] = rx, ry
        t_hist[fi], v_hist[fi], w_hist[fi], cte_hist[fi] = t, v, w, cte

        # Update Spatial
        path_arr = np.array(f["path"], dtype=float)
        if len(path_arr) >= 2:
            arts["corridor"].set_xy(corridor_polygon(path_arr, f.get("d_hard_eff", d_hard)))
        
        arts["path_line"].set_data(path_arr[:, 0], path_arr[:, 1])
        arts["robot_trace"].set_data(robot_x[:fi+1], robot_y[:fi+1])
        
        for key, source in [("ref_h", "ref"), ("pred_h", "pred")]:
            data = f.get(source, [])
            if data:
                arr = np.array(data, dtype=float)
                arts[key].set_data(arr[:, 0], arr[:, 1])
            else:
                arts[key].set_data([], [])

        proj = f.get("proj", [rx, ry])
        arts["proj"].set_data([proj[0]], [proj[1]])
        arts["robot_dot"].set_data([rx], [ry])
        arts["robot_dot"].set_color(C["solver_fail"] if not f.get("solver_ok", True) else C["robot_dot"])
        
        arts["heading"].set_offsets(np.array([[rx, ry]]))
        arts["heading"].set_UVC(np.cos(rth) * 0.15, np.sin(rth) * 0.15)
        arts["step_text"].set_text(f"t={t:03d}  CTE={cte:+.4f} m")

        # Update Time-series (Step 3: No dynamic rescaling here)
        arts["v_line"].set_data(t_hist[:fi+1], v_hist[:fi+1])
        arts["w_line"].set_data(t_hist[:fi+1], w_hist[:fi+1])
        arts["cte_line"].set_data(t_hist[:fi+1], cte_hist[:fi+1])

        # Update cursors
        arts["cursor_v"].set_xdata([t, t])
        arts["cursor_w"].set_xdata([t, t])
        arts["cursor_cte"].set_xdata([t, t])

        # Path-update events
        for ev_t in path_events:
            if ev_t not in drawn_events and ev_t <= t:
                kw = dict(color=C["path_update"], linewidth=0.9, linestyle=":", alpha=0.7)
                ax_v.axvline(ev_t, **kw)
                ax_w.axvline(ev_t, **kw)
                ax_cte.axvline(ev_t, **kw)
                drawn_events.add(ev_t)

        return []

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MPC Path Follower Visualizer")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--binary", help="C++ sim binary to execute")
    src.add_argument("--file", help="Existing JSONL data file to replay")
    ap.add_argument("--data", default="sim_data.jsonl")
    ap.add_argument("--save", help="Save animation to .mp4 or .gif")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    if args.binary:
        params, frames = load_from_binary(args.binary, data_path=args.data)
    else:
        params, frames = load_from_file(args.file)

    if not frames:
        sys.exit("[plot_sim] No frames loaded.")

    path_events = find_path_updates(frames)
    if args.save:
        matplotlib.use("Agg")

    fig, ax_xy, ax_v, ax_w, ax_cte = build_figure()

    # Step 2: Apply fixed limits once
    xmin, xmax, ymin, ymax = compute_xy_bounds(frames)
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    tb = compute_time_bounds(frames, params)
    for ax, key in [(ax_v, "v"), (ax_w, "w"), (ax_cte, "cte")]:
        ax.set_xlim(*tb["t"])
        ax.set_ylim(*tb[key])

    add_static_lines(ax_v, ax_w, ax_cte, params)
    arts = init_artists(ax_xy, ax_v, ax_w, ax_cte)
    update_fn = make_update(frames, arts, ax_v, ax_w, ax_cte, params, path_events)

    ani = animation.FuncAnimation(fig, update_fn, frames=len(frames), 
                                  interval=int(1000 / args.fps), blit=False, repeat=False)

    if args.save:
        print(f"[plot_sim] Saving → {args.save} …")
        writer = animation.FFMpegWriter(fps=args.fps) if args.save.endswith(".mp4") else animation.PillowWriter(fps=args.fps)
        ani.save(args.save, writer=writer, dpi=130, savefig_kwargs={"facecolor": C["bg_fig"]})
    else:
        plt.show()

if __name__ == "__main__":
    main()