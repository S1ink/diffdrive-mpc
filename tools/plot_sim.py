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
    "path":          "#44445a",
    "corridor_face": "#2244aa",
    "corridor_edge": "#4466cc",
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
    "cursor":        "#ffffff",
}

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Drawing Tool
# ─────────────────────────────────────────────────────────────────────────────
def draw_path_interactive():
    fig, ax = plt.subplots(figsize=(8, 8))
    pts = []

    line, = ax.plot([], [], "o-", color="black")
    ax.set_title("Left-click: Add Point | Right-click: Undo | Enter: Finish")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-5, 5)
    ax.grid(True)

    def redraw():
        if pts:
            arr = np.array(pts)
            line.set_data(arr[:,0], arr[:,1])
        else:
            line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button == 1:  # Left click
            pts.append((event.xdata, event.ydata))
        elif event.button == 3 and pts:  # Right click
            pts.pop()
        redraw()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return pts

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & Geometry bounds (Truncated for brevity, kept exactly same)
# ─────────────────────────────────────────────────────────────────────────────
def load_from_binary(binary_path: str, args):
    cmd = [binary_path]
    if args.scenario: cmd.extend(["--scenario", str(args.scenario)])
    if args.path: cmd.extend(["--path", args.path])
    
    print(f"[plot_sim] Running {' '.join(cmd)} …", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        sys.exit(f"Binary failed:\n{result.stderr}")

    lines = result.stdout.splitlines()
    if args.data:
        with open(args.data, "w") as f:
            f.write("\n".join(lines) + "\n")
    return _parse_lines(lines)

def load_from_file(filepath: str):
    with open(filepath) as f:
        return _parse_lines(f.read().splitlines())

def _parse_lines(lines):
    params, frames = {}, []
    for line in lines:
        if not line or line.startswith("#"): continue
        try:
            obj = json.loads(line.strip())
            if "params" in obj: params = obj["params"]
            else: frames.append(obj)
        except json.JSONDecodeError: continue
    return params, frames

def _offset_polyline(pts: np.ndarray, d: float):
    n = len(pts)
    normals = np.empty_like(pts)
    for i in range(n):
        seg = pts[1]-pts[0] if i==0 else (pts[-1]-pts[-2] if i==n-1 else pts[i+1]-pts[i-1])
        length = np.linalg.norm(seg)
        normals[i] = [0.0, 1.0] if length < 1e-9 else [-seg[1]/length, seg[0]/length]
    return pts + d * normals, pts - d * normals

def corridor_polygon(pts: np.ndarray, d: float):
    left, right = _offset_polyline(pts, d)
    return np.vstack([left, right[::-1]])

def compute_xy_bounds(frames, pad=0.8, min_span=4.0):
    xs, ys = [], []
    for f in frames:
        xs.append(f["robot"]["x"])
        ys.append(f["robot"]["y"])
        for source in ["path", "ref", "pred"]:
            for pt in f.get(source, []):
                xs.append(pt[0])
                ys.append(pt[1])
    if not xs: return (-min_span/2, min_span/2, -min_span/2, min_span/2)
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    
    if (x_max - x_min) < min_span:
        cx = (x_min + x_max) / 2
        x_min, x_max = cx - min_span / 2, cx + min_span / 2
    if (y_max - y_min) < min_span:
        cy = (y_min + y_max) / 2
        y_min, y_max = cy - min_span / 2, cy + min_span / 2
    return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)

# ─────────────────────────────────────────────────────────────────────────────
# Figure construction
# ─────────────────────────────────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=C["bg_fig"])
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1],
                           hspace=0.60, wspace=0.25, left=0.05, right=0.98)

    ax_xy  = fig.add_subplot(gs[0:4, 0])
    ax_v   = fig.add_subplot(gs[0, 1])
    ax_w   = fig.add_subplot(gs[1, 1])
    ax_cte = fig.add_subplot(gs[2, 1])
    ax_con = fig.add_subplot(gs[3, 1]) # Constraint activity subplot

    for ax in (ax_xy, ax_v, ax_w, ax_cte, ax_con):
        ax.set_facecolor(C["bg_ax"])
        ax.tick_params(colors=C["tick"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["spine"])

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("Spatial Trace (Colored by CTE)", color=C["title"], fontsize=10)
    ax_v.set_title("Forward velocity", color=C["title"], fontsize=9)
    ax_w.set_title("Angular velocity", color=C["title"], fontsize=9)
    ax_cte.set_title("Cross-track error", color=C["title"], fontsize=9)
    ax_con.set_title("Constraint Saturation Ratio", color=C["title"], fontsize=9)

    return fig, ax_xy, ax_v, ax_w, ax_cte, ax_con

def init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, params):
    arts = {}
    d_hard = params.get("d_hard", 0.1)
    
    arts["corridor"] = MplPolygon(np.zeros((4, 2)), closed=True, facecolor=C["corridor_face"], 
                                  alpha=0.20, edgecolor=C["corridor_edge"], linewidth=0.8, zorder=1)
    ax_xy.add_patch(arts["corridor"])
    arts["path_line"], = ax_xy.plot([], [], color=C["path"], linewidth=2.0, zorder=2)
    
    # Use scatter for robot trace to color by CTE
    arts["robot_trace"] = ax_xy.scatter([], [], c=[], cmap="coolwarm", s=10, 
                                        zorder=3, vmin=-d_hard, vmax=d_hard)
    
    arts["ref_h"], = ax_xy.plot([], [], "o--", color=C["ref_horizon"], markersize=3, alpha=0.8)
    arts["pred_h"], = ax_xy.plot([], [], "o-", color=C["pred_horizon"], markersize=3, alpha=0.85)
    arts["proj"], = ax_xy.plot([], [], "x", color=C["proj_pt"], markersize=8)
    arts["robot_dot"], = ax_xy.plot([], [], "o", color=C["robot_dot"], markersize=7, zorder=8)
    arts["heading"] = ax_xy.quiver(0, 0, 0, 0, color=C["robot_dot"], scale=5.0, scale_units="inches", width=0.005)

    arts["v_line"], = ax_v.plot([], [], color=C["v_line"])
    arts["w_line"], = ax_w.plot([], [], color=C["w_line"])
    arts["cte_line"], = ax_cte.plot([], [], color=C["cte_line"])
    
    # Constraint lines
    arts["con_w"], = ax_con.plot([], [], color=C["w_line"], label="|ω| / ω_max")
    arts["con_cte"], = ax_con.plot([], [], color=C["cte_line"], label="|CTE| / d_hard")
    ax_con.axhline(1.0, color=C["v_max"], linestyle="--", linewidth=1.0) # Saturation line
    ax_con.legend(loc="upper left", fontsize=7, facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    cursor_kw = dict(color=C["cursor"], alpha=0.3, linewidth=1, linestyle=":")
    arts["cursors"] = [ax.axvline(0, **cursor_kw) for ax in (ax_v, ax_w, ax_cte, ax_con)]

    return arts

# ─────────────────────────────────────────────────────────────────────────────
# Main Loop & Update
# ─────────────────────────────────────────────────────────────────────────────
def make_update(frames, arts, params):
    num_f = len(frames)
    rx_hist, ry_hist = np.zeros(num_f), np.zeros(num_f)
    t_hist, v_hist, w_hist, cte_hist = np.zeros(num_f), np.zeros(num_f), np.zeros(num_f), np.zeros(num_f)
    
    d_hard = params.get("d_hard", 0.10)
    w_max = params.get("omega_max", 1.2)

    def update(fi):
        f = frames[fi]
        t, rx, ry, rth = f["t"], f["robot"]["x"], f["robot"]["y"], f["robot"]["theta"]
        rx_hist[fi], ry_hist[fi], t_hist[fi] = rx, ry, t
        v_hist[fi], w_hist[fi], cte_hist[fi] = f["control"]["v"], f["control"]["omega"], f["cte"]

        path_arr = np.array(f["path"], dtype=float)
        if len(path_arr) >= 2:
            arts["corridor"].set_xy(corridor_polygon(path_arr, f.get("d_hard_eff", d_hard)))
        
        arts["path_line"].set_data(path_arr[:, 0], path_arr[:, 1])
        
        # Color trajectory by cross-track error
        arts["robot_trace"].set_offsets(np.c_[rx_hist[:fi+1], ry_hist[:fi+1]])
        arts["robot_trace"].set_array(cte_hist[:fi+1])
        
        for key, source in [("ref_h", "ref"), ("pred_h", "pred")]:
            arr = np.array(f.get(source, []), dtype=float)
            if len(arr): arts[key].set_data(arr[:, 0], arr[:, 1])

        proj = f.get("proj", [rx, ry])
        arts["proj"].set_data([proj[0]], [proj[1]])
        arts["robot_dot"].set_data([rx], [ry])
        arts["robot_dot"].set_color(C["solver_fail"] if not f.get("solver_ok", True) else C["robot_dot"])
        
        arts["heading"].set_offsets(np.array([[rx, ry]]))
        arts["heading"].set_UVC(np.cos(rth) * 0.15, np.sin(rth) * 0.15)

        arts["v_line"].set_data(t_hist[:fi+1], v_hist[:fi+1])
        arts["w_line"].set_data(t_hist[:fi+1], w_hist[:fi+1])
        arts["cte_line"].set_data(t_hist[:fi+1], cte_hist[:fi+1])
        
        # Constraint Saturation plotting
        arts["con_w"].set_data(t_hist[:fi+1], np.abs(w_hist[:fi+1]) / w_max)
        arts["con_cte"].set_data(t_hist[:fi+1], np.abs(cte_hist[:fi+1]) / d_hard)

        for cursor in arts["cursors"]: cursor.set_xdata([t, t])
        return []
    return update

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", help="Path to C++ sim binary")
    ap.add_argument("--file", help="JSONL file to replay")
    ap.add_argument("--data", default="sim_data.jsonl")
    ap.add_argument("--scenario", type=int, help="C++ Scenario ID")
    ap.add_argument("--path", type=str, help="Inject custom path file")
    ap.add_argument("--draw", action="store_true", help="Draw path interactively")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    # Interactive drawing mode
    if args.draw:
        pts = draw_path_interactive()
        if not pts:
            sys.exit("No points drawn.")
        
        out_file = "drawn_path.txt"
        with open(out_file, "w") as f:
            for pt in pts:
                f.write(f"{pt[0]} {pt[1]}\n")
        print(f"[plot_sim] Saved path to {out_file}")
        
        # Auto-inject the path into the binary if specified
        if args.binary:
            args.path = out_file
        else:
            sys.exit("Add --binary ./build/sim_test to immediately run what you drew.")

    if args.binary: params, frames = load_from_binary(args.binary, args)
    elif args.file: params, frames = load_from_file(args.file)
    else: sys.exit("Provide --binary or --file")

    fig, ax_xy, ax_v, ax_w, ax_cte, ax_con = build_figure()
    
    xmin, xmax, ymin, ymax = compute_xy_bounds(frames)
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    t_max = max([f["t"] for f in frames]) if frames else 100
    for ax in (ax_v, ax_w, ax_cte, ax_con): ax.set_xlim(0, t_max)
    
    ax_v.set_ylim(-0.05, params.get("v_max", 0.5) * 1.2)
    ax_w.set_ylim(-params.get("omega_max", 1.2)*1.2, params.get("omega_max", 1.2)*1.2)
    ax_cte.set_ylim(-params.get("d_hard", 0.1)*1.5, params.get("d_hard", 0.1)*1.5)
    ax_con.set_ylim(0, 1.5)

    arts = init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, params)
    update_fn = make_update(frames, arts, params)

    ani = animation.FuncAnimation(fig, update_fn, frames=len(frames), 
                                  interval=int(1000/args.fps), blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
