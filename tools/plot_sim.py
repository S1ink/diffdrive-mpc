#!/usr/bin/env python3
"""
plot_sim.py - MPC Path Follower Visualizer
=========================================

Runs the C++ sim binary (or reads a saved .jsonl file), then plays back
the results as a 20 Hz animation. Seven panels update together:

    +-------------------------+----------------------+
    |                         |   v(t)               |
    |   Spatial XY            +----------------------+
    |   (path, corridor,      |   omega(t)           |
    |    robot, horizons,     +----------------------+
    |    heading arrow)       |   Cross-track error  |
    |                         +----------------------+
    |                         |   Constraint ratios  |
    |                         +----------------------+
    |                         |   Solve time (ms)    |
    +-------------------------+----------------------+
    |   Run statistics                               |
    +------------------------------------------------+

Usage
-----
    python tools/plot_sim.py --binary ./build/sim_test
    python tools/plot_sim.py --file   sim_data.jsonl
    python tools/plot_sim.py --binary ./build/sim_test --save sim.mp4
    python tools/plot_sim.py --binary ./build/sim_test --noisy --prune
    python tools/plot_sim.py --binary ./build/sim_test --random
    python tools/plot_sim.py --binary ./build/sim_test --random --complexity 4 --stages 8
    python tools/plot_sim.py --binary ./build/sim_test --random --seed 42
    python tools/plot_sim.py --binary ./build/sim_test \\
                             --data  my_run.jsonl     \\
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
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch

# ---------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------
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
    "heading_arrow": "#00ddff",
    "v_line":        "#44aaff",
    "v_limit":       "#ff4444",  # v_max / v_min dashed lines
    "w_line":        "#ffaa33",
    "w_limit":       "#ff4444",  # ±omega_max dashed lines
    "cte_line":      "#cc44ff",
    "cte_band":      "#ff4444",
    "solver_fail":   "#ff2222",
    "solve_ok":      "#ffaa33",
    "solve_fail":    "#ff2222",
    "cursor":        "#ffffff",
    "recover_band":  "#ff220022",
    "stats_key":     "#9090bb",
    "stats_val":     "#e0e0ff",
    "stats_hi":      "#44aaff",
    "stage_line":    "#ffdd44",  # stage boundary markers
    "stage_marker":  "#ffdd44",
}

# ---------------------------------------------------------------------
# Interactive Drawing Tool
# ---------------------------------------------------------------------
def draw_path_interactive():
    fig, ax = plt.subplots(figsize=(8, 8))
    pts = []
    line, = ax.plot([], [], "o-", color="black")
    ax.set_title("Left-click: Add Point | Right-click: Undo | Enter: Finish")
    ax.set_xlim(-2, 10); ax.set_ylim(-5, 5); ax.grid(True)

    def redraw():
        if pts:
            arr = np.array(pts)
            line.set_data(arr[:, 0], arr[:, 1])
        else:
            line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button == 1: pts.append((event.xdata, event.ydata))
        elif event.button == 3 and pts: pts.pop()
        redraw()

    def on_key(event):
        if event.key == "enter": plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return pts

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_from_binary(binary_path: str, args):
    cmd = [binary_path]
    if args.scenario: cmd.extend(["--scenario", str(args.scenario)])
    if args.path:     cmd.extend(["--path", args.path])
    if args.config:   cmd.extend(["--config", args.config])
    if getattr(args, "noisy",      False): cmd.append("--noisy")
    if getattr(args, "prune",      False): cmd.append("--prune")
    if getattr(args, "random",     False): cmd.append("--random")
    if getattr(args, "complexity", None) is not None:
        cmd.extend(["--complexity", str(args.complexity)])
    if getattr(args, "stages",     None) is not None:
        cmd.extend(["--stages", str(args.stages)])
    if getattr(args, "seed",       None) is not None:
        cmd.extend(["--seed", str(args.seed)])
    print(f"[plot_sim] Running {' '.join(cmd)}...", file=sys.stderr)
    # stdout is captured for parsing; stderr is inherited so binary status
    # messages (path updates, goal reached, stuck, etc.) print to the terminal.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    stdout, _ = proc.communicate()
    if proc.returncode != 0:
        sys.exit(f"Binary exited with code {proc.returncode}")
    lines = stdout.splitlines()
    if args.data:
        with open(args.data, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return _parse_lines(lines)

def load_from_file(filepath: str):
    with open(filepath) as fh:
        return _parse_lines(fh.read().splitlines())

def _parse_lines(lines):
    params, frames = {}, []
    for line in lines:
        if not line or line.startswith("#"): continue
        try:
            obj = json.loads(line.strip())
            if "params" in obj: params = obj["params"]
            else: frames.append(obj)
        except json.JSONDecodeError:
            continue
    return params, frames

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def corridor_polygon(pts: np.ndarray, d: float, max_miter: float = 5.0):
    """Corridor polygon with proper miter joints at corners."""
    n = len(pts)
    if n < 2:
        return np.zeros((4, 2))

    segs    = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segs, axis=1, keepdims=True)
    dirs    = np.where(lengths > 1e-9, segs / lengths, np.array([[1.0, 0.0]]))
    normals = np.column_stack([-dirs[:, 1], dirs[:, 0]])  # left normal

    left, right = [], []
    for i in range(n):
        if i == 0:
            nv = normals[0]
        elif i == n - 1:
            nv = normals[-1]
        else:
            n1, n2 = normals[i - 1], normals[i]
            miter   = n1 + n2
            ml      = np.linalg.norm(miter)
            if ml < 1e-9:
                nv = n1
            else:
                miter_u = miter / ml
                denom   = float(np.dot(miter_u, n1))
                scale   = min(1.0 / denom, max_miter) if abs(denom) > 1e-6 else max_miter
                nv      = miter_u * scale
        left.append(pts[i] + d * nv)
        right.append(pts[i] - d * nv)

    return np.vstack([np.array(left), np.array(right)[::-1]])

def compute_xy_bounds(frames, pad=0.8, min_span=4.0):
    xs, ys = [], []
    for f in frames:
        xs.append(f["robot"]["x"]); ys.append(f["robot"]["y"])
        for src in ["path", "ref", "pred"]:
            for pt in f.get(src, []):
                xs.append(pt[0]); ys.append(pt[1])
    if not xs: return (-min_span / 2, min_span / 2, -min_span / 2, min_span / 2)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < min_span: cx = (x0 + x1) / 2; x0, x1 = cx - min_span / 2, cx + min_span / 2
    if y1 - y0 < min_span: cy = (y0 + y1) / 2; y0, y1 = cy - min_span / 2, cy + min_span / 2
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)

# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------
def build_figure():
    fig = plt.figure(figsize=(16, 11), facecolor=C["bg_fig"])

    # Outer grid: two rows.
    #   Row 0 (tall): XY map on left, 5 time-series panels on right.
    #   Row 1 (short): statistics bar spanning the full width.
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[5, 1],
        hspace=0.35,
        left=0.05, right=0.98, top=0.97, bottom=0.05,
    )

    # Top row: XY (left) + 5 panels (right)
    inner_top = gridspec.GridSpecFromSubplotSpec(
        5, 2,
        subplot_spec=outer[0],
        height_ratios=[1, 1, 1, 1, 1],
        hspace=0.75, wspace=0.28,
    )
    ax_xy    = fig.add_subplot(inner_top[0:5, 0])
    ax_v     = fig.add_subplot(inner_top[0, 1])
    ax_w     = fig.add_subplot(inner_top[1, 1])
    ax_cte   = fig.add_subplot(inner_top[2, 1])
    ax_con   = fig.add_subplot(inner_top[3, 1])
    ax_solve = fig.add_subplot(inner_top[4, 1])

    # Bottom row: stats spanning the full width
    ax_stats = fig.add_subplot(outer[1])

    for ax in (ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, ax_stats):
        ax.set_facecolor(C["bg_ax"])
        ax.tick_params(colors=C["tick"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["spine"])

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("Spatial Trace (Colored by CTE)",  color=C["title"], fontsize=10)
    ax_v.set_title("Forward velocity",                  color=C["title"], fontsize=9)
    ax_w.set_title("Angular velocity",                  color=C["title"], fontsize=9)
    ax_cte.set_title("Cross-track error",               color=C["title"], fontsize=9)
    ax_con.set_title("Constraint Saturation Ratio",     color=C["title"], fontsize=9)
    ax_solve.set_title("Solve time (ms)",               color=C["title"], fontsize=9)
    ax_stats.set_title("Run Statistics",                color=C["title"], fontsize=9)

    # Stats panel: turn off ticks/spines – we draw text manually
    ax_stats.set_xticks([]); ax_stats.set_yticks([])
    for sp in ax_stats.spines.values():
        sp.set_visible(False)

    return fig, ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, ax_stats

def init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, ax_stats, params):
    arts = {}
    d_hard  = params.get("d_hard", 0.1)
    v_max   = params.get("v_max", 2.5)
    v_min   = params.get("v_min", -0.1)
    w_max   = params.get("omega_max", 1.5)

    # ---- XY panel ------------------------------------------------------------
    arts["corridor"] = MplPolygon(
        np.zeros((4, 2)), closed=True,
        facecolor=C["corridor_face"], alpha=0.20,
        edgecolor=C["corridor_edge"], linewidth=0.8, zorder=1)
    ax_xy.add_patch(arts["corridor"])

    arts["path_line"],  = ax_xy.plot([], [], color=C["path"], linewidth=2.0, zorder=2)
    arts["robot_trace"] = ax_xy.scatter(
        [], [], c=[], cmap="coolwarm", s=10, zorder=3, vmin=-d_hard, vmax=d_hard)
    arts["ref_h"],  = ax_xy.plot([], [], "o--", color=C["ref_horizon"],  markersize=3, alpha=0.8)
    arts["pred_h"], = ax_xy.plot([], [], "o-",  color=C["pred_horizon"], markersize=3, alpha=0.85)
    arts["proj"],   = ax_xy.plot([], [], "x",   color=C["proj_pt"],      markersize=8)
    arts["robot_dot"], = ax_xy.plot([], [], "o", color=C["robot_dot"],   markersize=7, zorder=8)

    # Heading arrow: FancyArrowPatch supports set_positions() so it can be
    # reliably repositioned every frame.  The actual length is set in
    # make_update() once we know the data span.
    arts["heading"] = FancyArrowPatch(
        (0, 0), (1, 0),
        arrowstyle="->",
        mutation_scale=18,
        color=C["heading_arrow"],
        linewidth=1.8,
        zorder=9,
    )
    ax_xy.add_patch(arts["heading"])

    # ---- v(t) panel ----------------------------------------------------------
    arts["v_line"],  = ax_v.plot([], [], color=C["v_line"])
    ax_v.axhline(v_max, color=C["v_limit"], linestyle="--", linewidth=1.0,
                 label=f"v_max = {v_max:.2f}")
    ax_v.axhline(v_min, color=C["v_limit"], linestyle=":",  linewidth=1.0,
                 label=f"v_min = {v_min:.2f}")
    ax_v.legend(loc="upper right", fontsize=7,
                facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    # ---- omega(t) panel ------------------------------------------------------
    arts["w_line"],  = ax_w.plot([], [], color=C["w_line"])
    ax_w.axhline( w_max, color=C["w_limit"], linestyle="--", linewidth=1.0,
                  label=f"+ω_max = {w_max:.2f}")
    ax_w.axhline(-w_max, color=C["w_limit"], linestyle="--", linewidth=1.0,
                  label=f"−ω_max = {-w_max:.2f}")
    ax_w.legend(loc="upper right", fontsize=7,
                facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    # ---- CTE panel -----------------------------------------------------------
    arts["cte_line"], = ax_cte.plot([], [], color=C["cte_line"])

    # ---- Constraint saturation panel -----------------------------------------
    arts["con_w"],   = ax_con.plot([], [], color=C["w_line"],   label="|omega| / omega_max")
    arts["con_cte"], = ax_con.plot([], [], color=C["cte_line"], label="|CTE| / d_hard")
    ax_con.axhline(1.0, color=C["v_limit"], linestyle="--", linewidth=1.0)
    ax_con.legend(loc="upper left", fontsize=7,
                  facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    # ---- Solve-time panel ----------------------------------------------------
    arts["solve_line"],  = ax_solve.plot([], [], color=C["solve_ok"], linewidth=0.8)
    arts["solve_fails"]   = ax_solve.scatter(
        [], [], color=C["solve_fail"], s=18, zorder=5, label="recovery / fail")
    ax_solve.legend(loc="upper left", fontsize=7,
                    facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    # ---- Stats panel ---------------------------------------------------------
    # Full-width bar: 3 columns of (key, value) pairs.
    stat_defs = [
        ("Sim time",      "0.00 s"),
        ("Distance",      "0.00 m"),
        ("Avg velocity",  "0.00 m/s"),
        ("Max velocity",  "0.00 m/s"),
        ("Cur velocity",  "0.00 m/s"),
        ("Cur direction", "0.00°"),
        ("Avg |ω|",       "0.00 rad/s"),
        ("Max |ω|",       "0.00 rad/s"),
        ("Avg |CTE|",     "0.00 m"),
        ("Max |CTE|",     "0.00 m"),
        ("Stage",         "1"),
        ("Solver fails",  "0"),
    ]
    n_cols    = 3
    col_w     = 1.0 / n_cols
    n_rows    = -(-len(stat_defs) // n_cols)   # ceiling division
    row_h     = 0.88 / max(n_rows, 1)
    stats_texts = {}
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)
    for idx, (key, val) in enumerate(stat_defs):
        col   = idx % n_cols
        row   = idx // n_cols
        x_key = col * col_w + 0.01
        x_val = col * col_w + col_w - 0.015
        y     = 0.92 - row * row_h
        ax_stats.text(x_key, y, key + ":", transform=ax_stats.transAxes,
                      color=C["stats_key"], fontsize=8, va="top")
        stats_texts[key] = ax_stats.text(
            x_val, y, val, transform=ax_stats.transAxes,
            color=C["stats_val"], fontsize=8, va="top", ha="right",
            fontweight="bold")
    arts["stats"] = stats_texts

    # ---- Stage start markers (XY) -------------------------------------------
    # One scatter point per stage start position; revealed as playback passes
    # each transition.  Labelled with the stage number.
    arts["stage_markers"] = ax_xy.scatter(
        [], [], marker="D", s=45, color=C["stage_marker"],
        zorder=10, label="stage start")
    arts["stage_labels"]  = []   # populated in main() once pre is known

    # ---- Time cursors --------------------------------------------------------
    cursor_kw = dict(color=C["cursor"], alpha=0.3, linewidth=1, linestyle=":")
    arts["cursors"] = [
        ax.axvline(0, **cursor_kw)
        for ax in (ax_v, ax_w, ax_cte, ax_con, ax_solve)
    ]

    return arts

# ---------------------------------------------------------------------
# Pre-compute per-frame data arrays (done once, not per animation frame)
# ---------------------------------------------------------------------
def precompute(frames, params):
    n  = len(frames)
    d  = {
        "t":        np.zeros(n),
        "rx":       np.zeros(n),
        "ry":       np.zeros(n),
        "rth":      np.zeros(n),
        "v":        np.zeros(n),
        "omega":    np.zeros(n),
        "cte":      np.zeros(n),
        "solve_ms": np.zeros(n),
        "ok":       np.ones(n, dtype=bool),
    }
    for i, f in enumerate(frames):
        d["t"][i]        = f["t"]
        d["rx"][i]       = f["robot"]["x"]
        d["ry"][i]       = f["robot"]["y"]
        d["rth"][i]      = f["robot"]["theta"]
        d["v"][i]        = f["control"]["v"]
        d["omega"][i]    = f["control"]["omega"]
        d["cte"][i]      = f["cte"]
        d["solve_ms"][i] = f.get("solve_ms", 0.0)
        d["ok"][i]       = f.get("solver_ok", True)

    # Cumulative distance travelled
    dx          = np.diff(d["rx"], prepend=d["rx"][0])
    dy          = np.diff(d["ry"], prepend=d["ry"][0])
    d["dist"]   = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))

    # Stage transition detection: a new stage begins whenever the path end
    # point jumps by more than 1 m between consecutive frames.  This works
    # for both random-staged runs and the standard scenario-0 path update.
    stage_nums   = np.ones(n, dtype=int)
    stage_frames = [0]   # frame index where each stage begins
    prev_end     = None
    cur_stage    = 1
    for i, f in enumerate(frames):
        path_pts = f.get("path", [])
        if path_pts:
            end = np.array(path_pts[-1], dtype=float)
            if prev_end is not None:
                dist_jump = np.linalg.norm(end - prev_end)
                if dist_jump > 1.0:          # new path loaded
                    cur_stage += 1
                    stage_frames.append(i)
            prev_end = end
        stage_nums[i] = cur_stage
    d["stage"]        = stage_nums
    d["stage_frames"] = stage_frames   # list of frame indices

    return d

# ---------------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------------
def make_update(frames, arts, params, pre):
    d_hard = params.get("d_hard",     0.10)
    w_max  = params.get("omega_max",  1.50)

    # Arrow length: 5 % of the larger axis span, minimum 0.15 m.
    xy_span    = max(float(np.ptp(pre["rx"])), float(np.ptp(pre["ry"])), 3.0)
    arrow_len  = xy_span * 0.05

    def update(fi):
        f   = frames[fi]
        t   = pre["t"][fi]
        rx  = pre["rx"][fi]
        ry  = pre["ry"][fi]
        rth = pre["rth"][fi]

        sl = slice(0, fi + 1)

        # ---- XY panel --------------------------------------------------------
        path_arr = np.array(f["path"], dtype=float)
        if len(path_arr) >= 2:
            arts["corridor"].set_xy(
                corridor_polygon(path_arr, f.get("d_hard_eff", d_hard)))
        arts["path_line"].set_data(path_arr[:, 0], path_arr[:, 1])
        arts["robot_trace"].set_offsets(np.c_[pre["rx"][sl], pre["ry"][sl]])
        arts["robot_trace"].set_array(pre["cte"][sl])

        for key, src in [("ref_h", "ref"), ("pred_h", "pred")]:
            arr = np.array(f.get(src, []), dtype=float)
            if len(arr): arts[key].set_data(arr[:, 0], arr[:, 1])

        proj = f.get("proj", [rx, ry])
        arts["proj"].set_data([proj[0]], [proj[1]])
        arts["robot_dot"].set_data([rx], [ry])
        arts["robot_dot"].set_color(
            C["solver_fail"] if not f.get("solver_ok", True) else C["robot_dot"])

        # Heading arrow: tail at robot centre, tip in direction of theta.
        tip = (rx + arrow_len * np.cos(rth), ry + arrow_len * np.sin(rth))
        arts["heading"].set_positions((rx, ry), tip)

        # ---- Time-series panels ----------------------------------------------
        t_sl = pre["t"][sl]
        arts["v_line"].set_data(t_sl, pre["v"][sl])
        arts["w_line"].set_data(t_sl, pre["omega"][sl])
        arts["cte_line"].set_data(t_sl, pre["cte"][sl])
        arts["con_w"].set_data(t_sl, np.abs(pre["omega"][sl]) / w_max)
        arts["con_cte"].set_data(t_sl, np.abs(pre["cte"][sl]) / d_hard)

        # ---- Solve-time panel ------------------------------------------------
        arts["solve_line"].set_data(t_sl, pre["solve_ms"][sl])
        fail_mask = ~pre["ok"][sl]
        if fail_mask.any():
            arts["solve_fails"].set_offsets(
                np.c_[t_sl[fail_mask], pre["solve_ms"][sl][fail_mask]])
        else:
            arts["solve_fails"].set_offsets(np.empty((0, 2)))

        # ---- Time cursors ----------------------------------------------------
        for cursor in arts["cursors"]:
            cursor.set_xdata([t, t])

        # ---- Stage markers: show all transitions up to current frame ---------
        sf     = pre["stage_frames"]
        vis    = [i for i in sf if i <= fi]
        if vis:
            arts["stage_markers"].set_offsets(
                np.c_[pre["rx"][vis], pre["ry"][vis]])
        else:
            arts["stage_markers"].set_offsets(np.empty((0, 2)))

        # ---- Statistics panel ------------------------------------------------
        _update_stats(arts["stats"], pre, fi, rth)

        return []

    return update


def _update_stats(texts, pre, fi, rth):
    """Rewrite the live statistics text objects up to frame fi."""
    sl = slice(0, fi + 1)

    sim_time  = pre["t"][fi]
    dist      = pre["dist"][fi]
    v_sl      = pre["v"][sl]
    w_sl      = pre["omega"][sl]
    cte_sl    = np.abs(pre["cte"][sl])
    n_fails   = int((~pre["ok"][sl]).sum())
    cur_v     = pre["v"][fi]
    cur_deg   = float(np.degrees(rth) % 360)

    def _fmt(key, val):
        texts[key].set_text(val)

    _fmt("Sim time",      f"{sim_time:.2f} s")
    _fmt("Distance",      f"{dist:.3f} m")
    _fmt("Avg velocity",  f"{float(v_sl.mean()):.3f} m/s")
    _fmt("Max velocity",  f"{float(v_sl.max()):.3f} m/s")
    _fmt("Cur velocity",  f"{cur_v:.3f} m/s")
    _fmt("Cur direction", f"{cur_deg:.1f}°")
    _fmt("Avg |ω|",       f"{float(np.abs(w_sl).mean()):.3f} rad/s")
    _fmt("Max |ω|",       f"{float(np.abs(w_sl).max()):.3f} rad/s")
    _fmt("Avg |CTE|",     f"{float(cte_sl.mean()):.4f} m")
    _fmt("Max |CTE|",     f"{float(cte_sl.max()):.4f} m")
    stage_num = int(pre["stage"][fi])
    _fmt("Stage",        str(stage_num))
    _fmt("Solver fails",  str(n_fails))


# ---------------------------------------------------------------------
# Interactive playback  (pause / scrub)
# ---------------------------------------------------------------------
def run_interactive(fig, frames, arts, params, pre, fps):
    from matplotlib.widgets import Slider

    n         = len(frames)
    t_arr     = pre["t"]
    update_fn = make_update(frames, arts, params, pre)
    interval  = int(1000 / fps)

    fig.subplots_adjust(bottom=0.08)

    ax_sl  = fig.add_axes([0.08, 0.025, 0.88, 0.022], facecolor=C["bg_ax"])
    slider = Slider(ax_sl, '', 0, n - 1, valinit=0, valstep=1,
                    color=C["v_line"], initcolor="none")
    slider.label.set_color(C["tick"])      # type: ignore[attr-defined]
    slider.valtext.set_color(C["tick"])    # type: ignore[attr-defined]
    slider.valtext.set_fontsize(7)         # type: ignore[attr-defined]
    for spine in ax_sl.spines.values():
        spine.set_edgecolor(C["spine"])

    fig.text(0.5, 0.003,
             "SPACE: pause / play   Left/Right: step frame   HOME/END: jump to start/end",
             ha="center", va="bottom", fontsize=7, color=C["tick"])

    state = {"fi": 0, "playing": True}
    _busy = [False]

    def _draw(fi):
        if _busy[0]: return
        _busy[0] = True
        state["fi"] = fi
        slider.set_val(fi)
        slider.valtext.set_text(f"t = {t_arr[fi]:.2f} s  [{fi} / {n - 1}]")  # type: ignore[attr-defined]
        _busy[0] = False
        update_fn(fi)
        fig.canvas.draw_idle()

    def _tick():
        if not state["playing"]: return
        nfi = state["fi"] + 1
        if nfi >= n:
            state["playing"] = False
            return
        _draw(nfi)

    def _on_slider(val):
        if _busy[0]: return
        state["playing"] = False
        _draw(int(round(val)))

    def _on_key(event):
        k = event.key
        if k == " ":
            state["playing"] = not state["playing"]
            if state["playing"] and state["fi"] >= n - 1:
                state["fi"] = 0
        elif k == "left":
            state["playing"] = False
            _draw(max(0, state["fi"] - 1))
        elif k == "right":
            state["playing"] = False
            _draw(min(n - 1, state["fi"] + 1))
        elif k == "home":
            state["playing"] = False
            _draw(0)
        elif k == "end":
            state["playing"] = False
            _draw(n - 1)

    slider.on_changed(_on_slider)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(_tick)
    timer.start()

    _draw(0)
    plt.show()


# ---------------------------------------------------------------------
# Text stats summary (--stats mode, unchanged)
# ---------------------------------------------------------------------
def _print_stats(frames, params):
    ctes   = np.array([abs(f.get("cte", 0)) for f in frames])
    vs     = np.array([f["control"]["v"]     for f in frames])
    robots = np.array([[f["robot"]["x"], f["robot"]["y"]] for f in frames])
    n      = len(frames)
    t_end  = frames[-1].get("t", n * params.get("dt", 0.05))
    last   = frames[-1]

    # Total path length
    diffs    = np.diff(robots, axis=0)
    dist     = float(np.sum(np.linalg.norm(diffs, axis=1)))

    print(f"Frames      : {n}   sim_time : {t_end:.2f}s")
    print(f"Final       : x={last['robot']['x']:.4f}  y={last['robot']['y']:.4f}  "
          f"theta={last['robot']['theta']:.4f} rad  "
          f"({float(np.degrees(last['robot']['theta'])):.1f}°)")
    print(f"Distance    : {dist:.4f} m")
    print(f"CTE         : max={ctes.max():.4f}  mean={ctes.mean():.4f}  "
          f"p95={np.percentile(ctes, 95):.4f} m")
    print(f"Speed       : mean={vs.mean():.4f}  max={vs.max():.4f}  min={vs.min():.4f}  m/s")
    print(f"Avg speed   : {dist / t_end:.4f} m/s  (distance / sim_time)")

    corner = [f for f in frames if 1.6 <= f["robot"]["x"] <= 2.4 and f["robot"]["y"] <= 0.6]
    if corner:
        c_cte = [abs(f.get("cte", 0)) for f in corner]
        print(f"Corner      : {len(corner)} frames  max_cte={max(c_cte):.4f}m  "
              f"mean_v={sum(f['control']['v'] for f in corner)/len(corner):.4f} m/s")

    tail  = frames[int(n * 0.9):]
    t_vs  = [f["control"]["v"] for f in tail]
    print(f"Final 10%   : mean_v={sum(t_vs)/len(t_vs):.4f}  min_v={min(t_vs):.4f} m/s")
    solver_fails = sum(1 for f in frames if not f.get("solver_ok", True))
    print(f"Solver fails: {solver_fails}/{n}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",   help="Path to C++ sim binary")
    ap.add_argument("--file",     help="JSONL file to replay")
    ap.add_argument("--data",     default="sim_data.jsonl")
    ap.add_argument("--scenario", type=int, help="C++ scenario ID")
    ap.add_argument("--path",     type=str, help="Custom path file")
    ap.add_argument("--config",   type=str, help="MPC params config file (.cfg)")
    ap.add_argument("--noisy",      action="store_true", help="Enable sensor noise")
    ap.add_argument("--prune",      action="store_true", help="Enable path pruning")
    ap.add_argument("--random",     action="store_true", help="Random staged mode")
    ap.add_argument("--complexity", type=int, default=2, help="Path complexity 1-5 (random mode)")
    ap.add_argument("--stages",     type=int, default=5, help="Number of random stages")
    ap.add_argument("--seed",       type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--draw",     action="store_true", help="Draw path interactively")
    ap.add_argument("--fps",      type=int, default=20)
    ap.add_argument("--save",     help="Save animation to file (MP4 or GIF)")
    ap.add_argument("--stats",    action="store_true", help="Print text summary instead of animating")
    args = ap.parse_args()

    if args.draw:
        pts = draw_path_interactive()
        if not pts: sys.exit("No points drawn.")
        out_file = "drawn_path.txt"
        with open(out_file, "w") as fh:
            for pt in pts: fh.write(f"{pt[0]} {pt[1]}\n")
        print(f"[plot_sim] Saved path to {out_file}")
        if args.binary:
            args.path = out_file
        else:
            sys.exit("Add --binary ./build/sim_test to run the drawn path.")

    if   args.binary: params, frames = load_from_binary(args.binary, args)
    elif args.file:   params, frames = load_from_file(args.file)
    else:             sys.exit("Provide --binary or --file")

    if not frames:
        sys.exit("No frames parsed.")

    if args.stats:
        _print_stats(frames, params)
        return

    pre = precompute(frames, params)

    fig, ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, ax_stats = build_figure()

    xmin, xmax, ymin, ymax = compute_xy_bounds(frames)
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    t_max = float(pre["t"].max()) if len(pre["t"]) else 100.0
    for ax in (ax_v, ax_w, ax_cte, ax_con, ax_solve):
        ax.set_xlim(0, t_max)

    v_max = params.get("v_max", 2.5)
    v_min = params.get("v_min", -0.1)
    w_max = params.get("omega_max", 1.5)

    ax_v.set_ylim(v_min * 1.3 - 0.05, v_max * 1.25)
    ax_w.set_ylim(-w_max * 1.25, w_max * 1.25)
    ax_cte.set_ylim(-params.get("d_hard", 0.1) * 1.5,
                     params.get("d_hard", 0.1) * 1.5)
    ax_con.set_ylim(0, 1.5)

    solve_max = max(float(np.percentile(pre["solve_ms"], 95)) * 1.4, 5.0)
    ax_solve.set_ylim(0, solve_max)
    ax_solve.axhline(
        float(np.median(pre["solve_ms"])),
        color=C["spine"], linestyle=":", linewidth=0.8, alpha=0.6,
        label=f"median {np.median(pre['solve_ms']):.2f} ms")
    ax_solve.legend(loc="upper right", fontsize=7,
                    facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    arts = init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, ax_stats, params)

    # Draw static stage-boundary vertical lines on every time-series panel.
    # These are added once (not animated) so they sit underneath the live data.
    ts_axes = (ax_v, ax_w, ax_cte, ax_con, ax_solve)
    for si, fi in enumerate(pre["stage_frames"]):
        if fi == 0:
            continue   # don't mark the very start
        t_trans = float(pre["t"][fi])
        for ax in ts_axes:
            ax.axvline(t_trans, color=C["stage_line"], linewidth=0.9,
                       linestyle="--", alpha=0.6, zorder=0)
        # Label on the XY plot at the robot position at transition
        lbl = ax_xy.text(
            float(pre["rx"][fi]), float(pre["ry"][fi]),
            f" S{si + 1}", color=C["stage_marker"], fontsize=7,
            va="bottom", ha="left", zorder=11)
        arts["stage_labels"].append(lbl)

    if args.save:
        update_fn = make_update(frames, arts, params, pre)
        ani = animation.FuncAnimation(
            fig, update_fn, frames=len(frames),
            interval=int(1000 / args.fps), blit=False, repeat=False)
        print(f"[plot_sim] Saving animation to {args.save} ...", file=sys.stderr)
        ani.save(args.save, dpi=150)
        print("[plot_sim] Done.", file=sys.stderr)
    else:
        run_interactive(fig, frames, arts, params, pre, args.fps)


if __name__ == "__main__":
    main()
