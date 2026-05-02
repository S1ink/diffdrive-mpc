#!/usr/bin/env python3
"""
plot_sim.py  —  MPC Path Follower Visualizer
=============================================

Runs the C++ sim binary (or reads a saved .jsonl file), then plays back
the results as a 20 Hz animation.  Six panels update together:

    ┌─────────────────────────┬──────────────────────┐
    │                         │   v(t)               │
    │   Spatial XY            ├──────────────────────┤
    │   (path, corridor,      │   ω(t)               │
    │    robot, horizons)     ├──────────────────────┤
    │                         │   Cross-track error  │
    │                         ├──────────────────────┤
    │                         │   Constraint ratios  │
    │                         ├──────────────────────┤
    │                         │   Solve time (ms)    │
    └─────────────────────────┴──────────────────────┘

New in this version
-------------------
  • solve_ms panel  — OSQP wall-clock time per cycle.
      - Orange line for normal solves.
      - Red dots on frames where solver_ok=false (recovery / OSQP failure).
  • solver_fail vertical bands on v(t), ω(t), CTE panels.
  • Path segment pruning supported: the path array can shrink mid-run.

Usage
-----
    python tools/plot_sim.py --binary ./build/sim_test
    python tools/plot_sim.py --file   sim_data.jsonl
    python tools/plot_sim.py --binary ./build/sim_test --save sim.mp4
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
    "solve_ok":      "#ffaa33",
    "solve_fail":    "#ff2222",
    "cursor":        "#ffffff",
    "recover_band":  "#ff220022",  # semi-transparent red for recovery spans
}

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Drawing Tool
# ─────────────────────────────────────────────────────────────────────────────
def draw_path_interactive():
    fig, ax = plt.subplots(figsize=(8, 8))
    pts = []
    line, = ax.plot([], [], "o-", color="black")
    ax.set_title("Left-click: Add Point | Right-click: Undo | Enter: Finish")
    ax.set_xlim(-2, 10); ax.set_ylim(-5, 5); ax.grid(True)

    def redraw():
        if pts:
            arr = np.array(pts)
            line.set_data(arr[:,0], arr[:,1])
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

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
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

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def corridor_polygon(pts: np.ndarray, d: float, max_miter: float = 5.0):
    """Corridor polygon with proper miter joints at corners."""
    n = len(pts)
    if n < 2:
        return np.zeros((4, 2))

    segs = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segs, axis=1, keepdims=True)
    dirs = np.where(lengths > 1e-9, segs / lengths, np.array([[1.0, 0.0]]))
    # Left normal: rotate 90° CCW
    normals = np.column_stack([-dirs[:, 1], dirs[:, 0]])  # (n-1, 2)

    left, right = [], []
    for i in range(n):
        if i == 0:
            nv = normals[0]
        elif i == n - 1:
            nv = normals[-1]
        else:
            n1, n2 = normals[i - 1], normals[i]
            miter = n1 + n2
            ml = np.linalg.norm(miter)
            if ml < 1e-9:
                nv = n1  # 180° reversal
            else:
                miter_u = miter / ml
                denom = float(np.dot(miter_u, n1))
                scale = min(1.0 / denom, max_miter) if abs(denom) > 1e-6 else max_miter
                nv = miter_u * scale
        left.append(pts[i] + d * nv)
        right.append(pts[i] - d * nv)

    left = np.array(left)
    right = np.array(right)
    return np.vstack([left, right[::-1]])

def compute_xy_bounds(frames, pad=0.8, min_span=4.0):
    xs, ys = [], []
    for f in frames:
        xs.append(f["robot"]["x"]); ys.append(f["robot"]["y"])
        for src in ["path", "ref", "pred"]:
            for pt in f.get(src, []):
                xs.append(pt[0]); ys.append(pt[1])
    if not xs: return (-min_span/2, min_span/2, -min_span/2, min_span/2)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1-x0 < min_span: cx=(x0+x1)/2; x0,x1 = cx-min_span/2, cx+min_span/2
    if y1-y0 < min_span: cy=(y0+y1)/2; y0,y1 = cy-min_span/2, cy+min_span/2
    return (x0-pad, x1+pad, y0-pad, y1+pad)

# ─────────────────────────────────────────────────────────────────────────────
# Figure construction  (5 right-column panels)
# ─────────────────────────────────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(16, 10), facecolor=C["bg_fig"])
    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        height_ratios=[1, 1, 1, 1, 1],
        hspace=0.65, wspace=0.25,
        left=0.05, right=0.98, top=0.97, bottom=0.05,
    )
    ax_xy    = fig.add_subplot(gs[0:5, 0])
    ax_v     = fig.add_subplot(gs[0, 1])
    ax_w     = fig.add_subplot(gs[1, 1])
    ax_cte   = fig.add_subplot(gs[2, 1])
    ax_con   = fig.add_subplot(gs[3, 1])
    ax_solve = fig.add_subplot(gs[4, 1])

    for ax in (ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve):
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

    return fig, ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve

def init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, params):
    arts = {}
    d_hard = params.get("d_hard", 0.1)

    arts["corridor"] = MplPolygon(
        np.zeros((4, 2)), closed=True,
        facecolor=C["corridor_face"], alpha=0.20,
        edgecolor=C["corridor_edge"], linewidth=0.8, zorder=1)
    ax_xy.add_patch(arts["corridor"])

    arts["path_line"],  = ax_xy.plot([], [], color=C["path"], linewidth=2.0, zorder=2)
    arts["robot_trace"] = ax_xy.scatter(
        [], [], c=[], cmap="coolwarm", s=10, zorder=3, vmin=-d_hard, vmax=d_hard)
    arts["ref_h"],   = ax_xy.plot([], [], "o--", color=C["ref_horizon"],  markersize=3, alpha=0.8)
    arts["pred_h"],  = ax_xy.plot([], [], "o-",  color=C["pred_horizon"], markersize=3, alpha=0.85)
    arts["proj"],    = ax_xy.plot([], [], "x",   color=C["proj_pt"],      markersize=8)
    arts["robot_dot"], = ax_xy.plot([], [], "o", color=C["robot_dot"],    markersize=7, zorder=8)
    arts["heading"] = ax_xy.quiver(
        0, 0, 0, 0, color=C["robot_dot"],
        scale=5.0, scale_units="inches", width=0.005)

    arts["v_line"],   = ax_v.plot([],   [], color=C["v_line"])
    arts["w_line"],   = ax_w.plot([],   [], color=C["w_line"])
    arts["cte_line"], = ax_cte.plot([], [], color=C["cte_line"])

    arts["con_w"],   = ax_con.plot([], [], color=C["w_line"],   label="|ω| / ω_max")
    arts["con_cte"], = ax_con.plot([], [], color=C["cte_line"], label="|CTE| / d_hard")
    ax_con.axhline(1.0, color=C["v_max"], linestyle="--", linewidth=1.0)
    ax_con.legend(loc="upper left", fontsize=7,
                  facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    # Solve time: normal line + scatter dots coloured by solver_ok
    arts["solve_line"],   = ax_solve.plot([], [], color=C["solve_ok"], linewidth=0.8)
    arts["solve_fails"]   = ax_solve.scatter(
        [], [], color=C["solve_fail"], s=18, zorder=5, label="recovery / fail")
    ax_solve.legend(loc="upper left", fontsize=7,
                    facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    cursor_kw = dict(color=C["cursor"], alpha=0.3, linewidth=1, linestyle=":")
    arts["cursors"] = [
        ax.axvline(0, **cursor_kw)
        for ax in (ax_v, ax_w, ax_cte, ax_con, ax_solve)
    ]

    return arts

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute per-frame data arrays (done once, not per animation frame)
# ─────────────────────────────────────────────────────────────────────────────
def precompute(frames, params):
    n = len(frames)
    d = {
        "t":        np.zeros(n),
        "rx":       np.zeros(n),
        "ry":       np.zeros(n),
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
        d["v"][i]        = f["control"]["v"]
        d["omega"][i]    = f["control"]["omega"]
        d["cte"][i]      = f["cte"]
        d["solve_ms"][i] = f.get("solve_ms", 0.0)
        d["ok"][i]       = f.get("solver_ok", True)
    return d

# ─────────────────────────────────────────────────────────────────────────────
# Animation update
# ─────────────────────────────────────────────────────────────────────────────
def make_update(frames, arts, params, pre):
    d_hard = params.get("d_hard", 0.10)
    w_max  = params.get("omega_max", 1.2)

    def update(fi):
        f  = frames[fi]
        t  = pre["t"][fi]
        rx = pre["rx"][fi]
        ry = pre["ry"][fi]
        rth = f["robot"]["theta"]

        sl = slice(0, fi + 1)

        # XY panel
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
        arts["heading"].set_offsets(np.array([[rx, ry]]))
        arts["heading"].set_UVC(np.cos(rth) * 0.15, np.sin(rth) * 0.15)

        # Time-series panels
        t_sl = pre["t"][sl]
        arts["v_line"].set_data(t_sl, pre["v"][sl])
        arts["w_line"].set_data(t_sl, pre["omega"][sl])
        arts["cte_line"].set_data(t_sl, pre["cte"][sl])
        arts["con_w"].set_data(t_sl, np.abs(pre["omega"][sl]) / w_max)
        arts["con_cte"].set_data(t_sl, np.abs(pre["cte"][sl]) / d_hard)

        # Solve-time panel
        arts["solve_line"].set_data(t_sl, pre["solve_ms"][sl])
        # Scatter dots for recovery / failed solves
        fail_mask = ~pre["ok"][sl]
        if fail_mask.any():
            arts["solve_fails"].set_offsets(
                np.c_[t_sl[fail_mask], pre["solve_ms"][sl][fail_mask]])
        else:
            arts["solve_fails"].set_offsets(np.empty((0, 2)))

        for cursor in arts["cursors"]:
            cursor.set_xdata([t, t])

        return []
    return update

# ─────────────────────────────────────────────────────────────────────────────
# Interactive playback  (pause / scrub)
# ─────────────────────────────────────────────────────────────────────────────
def run_interactive(fig, frames, arts, params, pre, fps):
    from matplotlib.widgets import Slider

    n          = len(frames)
    t_arr      = pre["t"]
    update_fn  = make_update(frames, arts, params, pre)
    interval   = int(1000 / fps)

    # Push subplots up to make room for the slider
    fig.subplots_adjust(bottom=0.08)

    ax_sl = fig.add_axes([0.08, 0.025, 0.88, 0.022], facecolor=C["bg_ax"])
    slider = Slider(ax_sl, '', 0, n - 1, valinit=0, valstep=1,
                    color=C["v_line"], initcolor="none")
    slider.label.set_color(C["tick"])      # type: ignore[attr-defined]
    slider.valtext.set_color(C["tick"])    # type: ignore[attr-defined]
    slider.valtext.set_fontsize(7)         # type: ignore[attr-defined]
    for spine in ax_sl.spines.values():
        spine.set_edgecolor(C["spine"])

    fig.text(0.5, 0.003,
             "SPACE: pause / play   ←/→: step frame   HOME/END: jump to start/end",
             ha="center", va="bottom", fontsize=7, color=C["tick"])

    state    = {"fi": 0, "playing": True}
    _busy    = [False]

    def _draw(fi):
        if _busy[0]:
            return
        _busy[0] = True
        state["fi"] = fi
        slider.set_val(fi)
        slider.valtext.set_text(f"t = {t_arr[fi]:.2f} s  [{fi} / {n - 1}]")  # type: ignore[attr-defined]
        _busy[0] = False
        update_fn(fi)
        fig.canvas.draw_idle()

    def _tick():
        if not state["playing"]:
            return
        nfi = state["fi"] + 1
        if nfi >= n:
            state["playing"] = False
            return
        _draw(nfi)

    def _on_slider(val):
        if _busy[0]:
            return
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def _print_stats(frames, params):
    import numpy as np
    ctes   = np.array([abs(f.get("cte", 0)) for f in frames])
    vs     = np.array([f["control"]["v"]     for f in frames])
    robots = np.array([[f["robot"]["x"], f["robot"]["y"]] for f in frames])
    n      = len(frames)
    t_end  = frames[-1].get("t", n * params.get("dt", 0.05))
    last   = frames[-1]

    print(f"Frames : {n}   sim_time : {t_end:.2f}s")
    print(f"Final  : x={last['robot']['x']:.4f}  y={last['robot']['y']:.4f}  "
          f"θ={last['robot']['theta']:.4f}  v={last['control']['v']:.4f}")
    print(f"CTE    : max={ctes.max():.4f}  mean={ctes.mean():.4f}  "
          f"p95={np.percentile(ctes,95):.4f}  m")
    print(f"Speed  : mean={vs.mean():.4f}  max={vs.max():.4f}  min={vs.min():.4f}  m/s")

    # Corner analysis: frames where robot is near x=2 (corner zone)
    corner = [f for f in frames if 1.6 <= f["robot"]["x"] <= 2.4 and f["robot"]["y"] <= 0.6]
    if corner:
        c_cte = [abs(f.get("cte", 0)) for f in corner]
        print(f"Corner : {len(corner)} frames  max_cte={max(c_cte):.4f}m  "
              f"mean_v={sum(f['control']['v'] for f in corner)/len(corner):.4f} m/s")

    # Last 10% of frames for goal approach
    tail = frames[int(n * 0.9):]
    t_vs = [f["control"]["v"] for f in tail]
    print(f"Final10%: mean_v={sum(t_vs)/len(t_vs):.4f}  min_v={min(t_vs):.4f} m/s")
    solver_fails = sum(1 for f in frames if not f.get("solver_ok", True))
    print(f"Solver fails: {solver_fails}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",   help="Path to C++ sim binary")
    ap.add_argument("--file",     help="JSONL file to replay")
    ap.add_argument("--data",     default="sim_data.jsonl")
    ap.add_argument("--scenario", type=int, help="C++ scenario ID")
    ap.add_argument("--path",     type=str, help="Custom path file")
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

    fig, ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve = build_figure()

    xmin, xmax, ymin, ymax = compute_xy_bounds(frames)
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    t_max = float(pre["t"].max()) if len(pre["t"]) else 100.0
    for ax in (ax_v, ax_w, ax_cte, ax_con, ax_solve):
        ax.set_xlim(0, t_max)

    ax_v.set_ylim(-0.05,  params.get("v_max",     0.5)  * 1.2)
    ax_w.set_ylim(-params.get("omega_max", 1.2)*1.2,
                   params.get("omega_max", 1.2)*1.2)
    ax_cte.set_ylim(-params.get("d_hard", 0.1)*1.5,
                     params.get("d_hard", 0.1)*1.5)
    ax_con.set_ylim(0, 1.5)

    # Solve-time y-axis: 95th-percentile cap so occasional spikes don't
    # compress the scale; full range visible via zoom.
    solve_max = max(float(np.percentile(pre["solve_ms"], 95)) * 1.4, 5.0)
    ax_solve.set_ylim(0, solve_max)
    ax_solve.axhline(
        float(np.median(pre["solve_ms"])),
        color=C["spine"], linestyle=":", linewidth=0.8, alpha=0.6,
        label=f"median {np.median(pre['solve_ms']):.2f} ms")
    ax_solve.legend(loc="upper right", fontsize=7,
                    facecolor=C["bg_ax"], edgecolor=C["spine"], labelcolor=C["tick"])

    arts = init_artists(ax_xy, ax_v, ax_w, ax_cte, ax_con, ax_solve, params)

    if args.save:
        update_fn = make_update(frames, arts, params, pre)
        ani = animation.FuncAnimation(
            fig, update_fn, frames=len(frames),
            interval=int(1000 / args.fps), blit=False, repeat=False)
        print(f"[plot_sim] Saving animation to {args.save} …", file=sys.stderr)
        ani.save(args.save, dpi=150)
        print("[plot_sim] Done.", file=sys.stderr)
    else:
        run_interactive(fig, frames, arts, params, pre, args.fps)

if __name__ == "__main__":
    main()
