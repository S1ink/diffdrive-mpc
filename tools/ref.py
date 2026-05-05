#!/usr/bin/env python3
"""
MPC Reference Generator - Interactive Debug Tool
=================================================
Requirement mapping:

  Req 1  Dynamic path / stateless design
         Hash-based path change detection re-derives seg_idx via bisector
         traversal from 0 on every path change.  v_cur (robot speed) is
         preserved across path changes (it's robot state, not path state)
         and reset only on explicit teleport.  Pure polyline refs mean no
         per-frame blending state is needed; the optimizer owns recovery.

  Req 2  Polyline-only sampling + corridor subgoal
         References are sampled exclusively along polyline arc length.
         alpha (CTE / corridor ratio) is exposed as a diagnostic / MPC cost
         weight; NO recovery geometry is injected into the refs themselves
         (this was the key architectural bug in the original).

    Req 3  Kinematic velocity profile
                 - Forward integration bounded by a_max and v_max
                 - Look-ahead braking: for every junction and the end-stop, compute
                     the maximum speed NOW that still allows braking to the constraint
                     speed on time:  v_cap = sqrt(v_ev^2 + 2*a_max*ds)
                 - Junction speed limit: v_lim = omega_max * ds_avg / Delta_theta
                     (from omega = v * Delta_theta / ds_avg <= omega_max)
                 - Profile reaches exactly v = 0 at the final waypoint

  Req 4  Bisector-based projection
         Each internal vertex gets a forward angle-bisector plane.
         Segment index advances ONLY when the robot crosses the bisector at
         the next junction - eliminates both backward jumps and premature
         transitions.  Projection is performed on the CURRENT segment only.

Controls
--------
  SHIFT + Left-click    add waypoint
  SHIFT + Right-click   remove last waypoint
  SHIFT + drag robot    click within 0.4 units of robot dot, then drag
  SHIFT + Arrow keys    translate robot
  SHIFT + A / D         rotate heading
  R                     reset robot to origin
  C                     clear path
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
import hashlib


# ======================================================================
# Global state
# ======================================================================

path  = []                           # [(x, y, ...)]
robot = np.array([0.0, 0.0, 0.0])   # [x, y, theta]  theta in radians

_st = dict(
    seg_idx   = 0,     # current segment (0-based)
    v_cur     = 0.0,   # speed estimate (m/s); from odometry in real use
    path_hash = None,
)

params = dict(
    v_max     = 1.0,   # m/s
    a_max     = 0.5,   # m/s^2
    omega_max = 1.5,   # rad/s  - governs junction slowdown
    dt        = 0.05,  # s
    horizon   = 20,    # steps
    corridor  = 0.3,   # m
)

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
    # ======================================================================
    # Sliders
    # ======================================================================

    def _sl(rect, label, lo, hi, val):
        return Slider(fig.add_axes(rect), label, lo, hi, valinit=val, color='steelblue')

# ======================================================================
# Figure layout
# ======================================================================

fig   = plt.figure(figsize=(14, 8))
def make_bisectors(poly):
    """
    Forward angle bisectors at each internal vertex i  (1 <= i <= N-2).

    bisector = normalise(d_in_unit + d_out_unit)

    For degenerate U-turns (antiparallel segments the sum is ~0) we fall
    back to the left-perpendicular of the incoming segment so that the
    half-space logic still works.

    Returns dict {vertex_idx: unit_vector}.
    """
    b = {}
    for i in range(1, len(poly) - 1):
        d_in  = np.asarray(poly[i],   float) - np.asarray(poly[i-1], float)
        d_out = np.asarray(poly[i+1], float) - np.asarray(poly[i],   float)
        d_in  /= np.linalg.norm(d_in)  + 1e-9
        d_out /= np.linalg.norm(d_out) + 1e-9
        bv = d_in + d_out
        nb = np.linalg.norm(bv)
        b[i] = bv / nb if nb > 1e-6 else np.array([-d_in[1], d_in[0]])
    return b


def project_seg(p, a, b):
    """Project point p onto segment a->b.  Returns (proj, t in [0,1], dist)."""
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-12:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t    = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
    proj = a + t * ab
    return proj, t, float(np.linalg.norm(p - proj))


def advance_seg(pos, poly, idx, bis):
    """
    Advance segment index while pos has crossed the bisector plane at the
    NEXT junction vertex.  Never moves backward.
    """
    while idx < len(poly) - 2:
        junc = np.asarray(poly[idx + 1], float)
        bv   = bis.get(idx + 1)
        if bv is None or np.dot(pos - junc, bv) < 0.0:
            break
        idx += 1
    return idx


def sample_poly(poly, cum, s0, arc_ds):
    """
    Sample polyline at absolute arc lengths {s0 + d | d in arc_ds}.
    Results are clamped to the end of the path.
    """
    total = float(cum[-1])
    pts   = []
    for d in arc_ds:
        s   = min(float(s0) + float(d), total)
        i   = max(0, min(int(np.searchsorted(cum, s, side='right')) - 1,
                        len(poly) - 2))
        L   = float(cum[i + 1] - cum[i])
        t   = np.clip((s - float(cum[i])) / max(L, 1e-9), 0.0, 1.0)
        pts.append(np.asarray(poly[i], float) +
                   t * (np.asarray(poly[i+1], float) - np.asarray(poly[i], float)))
    return pts


# ======================================================================
# Velocity profile
# ======================================================================

def vel_profile(poly, cum, segl, s0, v0, p):
    """
    Kinematically-feasible velocity profile for `horizon` steps.

    Returns (velocities, arc_distances, events).
    `events` is [(arc_length, speed_limit), ...] for visualisation.

    Constraint collection
    ---------------------
    End-of-path stop  ->  (total, 0.0)
    Each junction i with |Delta_theta| > 0.05 rad:
        v_lim = omega_max * ds_avg / Delta_theta   (from omega = v * Delta_theta / ds_avg <= omega_max)

    Forward integration with look-ahead braking
    --------------------------------------------
    At each step, find the tightest speed cap that allows reaching every
    upcoming constraint:
        v_cap = min_events  sqrt(v_ev^2 + 2*a_max*(s_ev - s0 - s))
    Then advance velocity toward v_cap within +/- a_max*dt.
    """
    dt, vmax, amax, omax, N = (
        p['dt'], p['v_max'], p['a_max'], p['omega_max'], p['horizon']
    )
    total = float(cum[-1])

    # Guard: robot already at or past end
    if s0 >= total - 1e-6:
        return [0.0] * N, [0.0] * N, [(total, 0.0)]

    # -- Constraint events -----------------------------------------------
    events = [(total, 0.0)]   # end-stop

    for i in range(1, len(poly) - 1):
        s_j = float(cum[i])
        if s_j <= s0 + 1e-6:
            continue
        d_in  = np.asarray(poly[i],   float) - np.asarray(poly[i-1], float)
        d_out = np.asarray(poly[i+1], float) - np.asarray(poly[i],   float)
        ni, no = np.linalg.norm(d_in), np.linalg.norm(d_out)
        if ni < 1e-9 or no < 1e-9:
            continue
        cos_a = np.clip(np.dot(d_in / ni, d_out / no), -1.0, 1.0)
        angle = float(np.arccos(cos_a))
        if angle < 0.05:          # < ~3 deg, negligible
            continue
        ds_avg = 0.5 * (float(segl[i-1]) + float(segl[i]))
        v_lim  = float(np.clip(omax * ds_avg / (angle + 1e-9), 0.0, vmax))
        events.append((s_j, v_lim))

    # -- Forward integration --------------------------------------------
    velocities = []
    arc_dists  = []
    v = float(np.clip(v0, 0.0, vmax))
    s = 0.0

    for _ in range(N):
        v_cap = vmax
        for s_ev, v_ev in events:
            ds = (s_ev - s0) - s
            if ds >= 0:
                # Max speed now to brake to v_ev over distance ds
                v_cap = min(v_cap,
                            float(np.sqrt(max(0.0, v_ev**2 + 2.0 * amax * ds))))

        v = float(np.clip(v_cap, v - amax * dt, v + amax * dt))
        v = max(v, 0.0)

        s += v * dt
        velocities.append(v)
        arc_dists.append(s)

    return velocities, arc_dists, events


# ======================================================================
# Reference generation  (main entry point)
# ======================================================================

def generate_refs():
    if len(path) < 2:
        return {}

    h = hashlib.md5(str(path).encode()).hexdigest()
    if h != _st['path_hash']:
        # Path changed: re-derive current segment via full bisector traversal.
        # v_cur is intentionally preserved - it is robot state, not path state.
        bis_tmp = make_bisectors(path)
        _st.update(
            seg_idx   = advance_seg(robot[:2], path, 0, bis_tmp),
            path_hash = h,
        )

    pos       = robot[:2].copy()
    cum, segl = poly_cum(path)
    bis       = make_bisectors(path)

    # -- Bisector-gated segment advance ---------------------------------
    _st['seg_idx'] = advance_seg(pos, path, _st['seg_idx'], bis)
    seg = _st['seg_idx']

    # -- Project onto CURRENT segment only -------------------------------
    a = np.asarray(path[seg],     float)
    b = np.asarray(path[seg + 1], float)
    proj_pt, t_seg, cte = project_seg(pos, a, b)
    s0 = float(cum[seg]) + t_seg * float(segl[seg])

    # -- Kinematic velocity profile -------------------------------------
    vs, arc_ds, events = vel_profile(path, cum, segl, s0, _st['v_cur'], params)
    _st['v_cur'] = vs[0] if vs else 0.0   # propagate speed estimate

    # -- Sample references ONLY on polyline geometry (Req 2) -------------
    #    The corridor is exposed as alpha for the optimizer to use as a cost weight.
    #    No recovery geometry is blended into the refs here.
    refs  = sample_poly(path, cum, s0, arc_ds)
    alpha = float(np.clip(1.0 - cte / max(params['corridor'], 1e-9), 0.0, 1.0))

    return dict(
        refs   = refs,
        proj   = proj_pt,
        cte    = cte,
        s0     = s0,
        vs     = vs,
        arc_ds = arc_ds,
        alpha  = alpha,
        seg    = seg,
        bis    = bis,
        cum    = cum,
        segl   = segl,
        events = events,
    )

# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------
fig   = plt.figure(figsize=(14, 8))
ax_xy = fig.add_axes([0.04, 0.28, 0.56, 0.68])
ax_v  = fig.add_axes([0.65, 0.54, 0.33, 0.40])
ax_al = fig.add_axes([0.65, 0.28, 0.33, 0.18])

# -- XY plot static artists ---------------------------------------------
path_ln,  = ax_xy.plot([], [], 'k-',   lw=1.5,  zorder=2, label='path')
seg_ln,   = ax_xy.plot([], [], '-',    color='limegreen', lw=5, alpha=0.45,
                        zorder=3, label='active seg')
ref_ln,   = ax_xy.plot([], [], 'o--',  color='darkorange', ms=5, alpha=0.9,
                        zorder=5, label='refs')
rob_dot,  = ax_xy.plot([], [], 'o',    color='royalblue', ms=11, zorder=7)
proj_dot, = ax_xy.plot([], [], 'rx',   ms=9, mew=2.5, zorder=6, label='projection')
head_q    = ax_xy.quiver(0, 0, 1, 0,   color='royalblue', scale=6, zorder=8)
corr_c    = Circle((0, 0), 0.3, fill=False, ls='--', ec='dimgray', lw=1.2, zorder=4)
ax_xy.add_patch(corr_c)

ax_xy.set_aspect('equal')
ax_xy.set_xlim(-6, 6); ax_xy.set_ylim(-6, 6)
ax_xy.grid(True, alpha=0.25)
ax_xy.legend(loc='upper left', fontsize=8, framealpha=0.8)

# -- Velocity profile plot ---------------------------------------------
v_ln, = ax_v.plot([], [], 'royalblue', lw=2, label='v(t)')
ax_v.set_title('Velocity Profile',   fontsize=9)
ax_v.set_ylabel('v  (m/s)',           fontsize=8)
ax_v.set_xlabel('horizon step',       fontsize=8)
ax_v.tick_params(labelsize=7)
ax_v.legend(fontsize=8)

# -- Corridor alpha plot -----------------------------------------------
al_ln, = ax_al.plot([], [], color='seagreen', lw=2)
ax_al.axhline(1.0, color='gray', lw=0.8, ls='--')
ax_al.set_title('Corridor alpha  (1 = on path,  0 = at/beyond corridor edge)',
                fontsize=8)
ax_al.set_ylim(0.0, 1.15)
ax_al.tick_params(labelsize=7)

# Dynamic per-frame artists (cleared and rebuilt each update)
_bis_arts   = []   # bisector lines in XY
_event_arts = []   # event markers in velocity plot
# ---------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------
def _sl(rect, label, lo, hi, val):
    return Slider(fig.add_axes(rect), label, lo, hi, valinit=val, color='steelblue')

sl_v = _sl([0.06, 0.19, 0.22, 0.025], 'v_max',     0.1, 3.0, params['v_max'])
sl_a = _sl([0.06, 0.14, 0.22, 0.025], 'a_max',     0.1, 3.0, params['a_max'])
sl_o = _sl([0.06, 0.09, 0.22, 0.025], 'omega_max', 0.1, 5.0, params['omega_max'])
sl_h = _sl([0.36, 0.19, 0.22, 0.025], 'horizon',   5,   60,  params['horizon'])
sl_c = _sl([0.36, 0.14, 0.22, 0.025], 'corridor',  0.05, 2.0, params['corridor'])

sl_v.on_changed(lambda v: params.update(v_max=v))
sl_a.on_changed(lambda v: params.update(a_max=v))
sl_o.on_changed(lambda v: params.update(omega_max=v))
sl_h.on_changed(lambda v: params.update(horizon=int(v)))
sl_c.on_changed(lambda v: params.update(corridor=v))

# ---------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------

_drag = [False]

def on_click(e):
    if e.inaxes != ax_xy or e.xdata is None: return
    if 'shift' not in (e.key or ''):         return
    if e.button == 1:
        if np.hypot(e.xdata - robot[0], e.ydata - robot[1]) < 0.4:
            _drag[0] = True
        else:
            path.append((float(e.xdata), float(e.ydata)))
    elif e.button == 3 and path:
        path.pop()

def on_release(_):
    _drag[0] = False

def on_move(e):
    if _drag[0] and e.inaxes == ax_xy and e.xdata is not None:
        robot[0], robot[1] = e.xdata, e.ydata
        _st['v_cur'] = 0.0   # teleported - reset speed

def on_key(e):
    step, ang = 0.08, 0.08
    k  = e.key or ''
    ku = k.upper()

    if ku == 'R':
        robot[:] = [0, 0, 0]; _st['v_cur'] = 0.0
    if ku == 'C':
        path.clear(); _st.update(seg_idx=0, path_hash=None, v_cur=0.0)

    if 'shift' not in k.lower(): return

    moved = False
    if 'left'  in k: robot[0] -= step; moved = True
    if 'right' in k: robot[0] += step; moved = True
    if 'up'    in k: robot[1] += step; moved = True
    if 'down'  in k: robot[1] -= step; moved = True
    if 'a'     in k: robot[2] += ang
    if 'd'     in k: robot[2] -= ang
    if moved:
        _st['v_cur'] = 0.0   # teleported

fig.canvas.mpl_connect('button_press_event',   on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event',  on_move)
fig.canvas.mpl_connect('key_press_event',      on_key)

# ---------------------------------------------------------------------
# Update / render
# ---------------------------------------------------------------------

def _clear_dynamic():
    for a in _bis_arts:   a.remove()
    for a in _event_arts: a.remove()
    _bis_arts.clear()
    _event_arts.clear()


def update():
    _clear_dynamic()
    res = generate_refs()

    # -- Path polyline --------------------------------------------------
    if len(path) >= 2:
        arr = np.asarray(path, float)
        path_ln.set_data(arr[:, 0], arr[:, 1])
    else:
        path_ln.set_data([], [])

    # -- Robot dot + heading -------------------------------------------
    rob_dot.set_data([robot[0]], [robot[1]])
    head_q.set_offsets([[robot[0], robot[1]]])
    head_q.set_UVC(np.cos(robot[2]), np.sin(robot[2]))

    if not res:
        seg_ln.set_data([], [])
        ref_ln.set_data([], [])
        proj_dot.set_data([], [])
        v_ln.set_data([], [])
        al_ln.set_data([], [])
        ax_xy.set_title('Place >= 2 waypoints  (SHIFT + click)', fontsize=10)
        fig.canvas.draw_idle()
        return

    # -- Active segment (highlighted) ---------------------------------
    seg = res['seg']
    sa  = np.asarray(path[seg],     float)
    sb  = np.asarray(path[seg + 1], float)
    seg_ln.set_data([sa[0], sb[0]], [sa[1], sb[1]])

    # -- Bisector planes (magenta tick marks at each junction) ---------
    for idx, bv in res['bis'].items():
        if idx >= len(path): continue
        junc = np.asarray(path[idx], float)
        perp = np.array([-bv[1], bv[0]])
        L    = 0.22
        ln,  = ax_xy.plot(
            [junc[0] - L*perp[0], junc[0] + L*perp[0]],
            [junc[1] - L*perp[1], junc[1] + L*perp[1]],
            'm-', lw=1.5, alpha=0.70, zorder=3
        )
        _bis_arts.append(ln)

    # -- Reference points ----------------------------------------------
    if res['refs']:
        r = np.array(res['refs'])
        ref_ln.set_data(r[:, 0], r[:, 1])
    else:
        ref_ln.set_data([], [])

    # -- Projection + corridor circle ----------------------------------
    px, py = res['proj']
    proj_dot.set_data([px], [py])
    corr_c.center = (px, py)
    corr_c.radius = params['corridor']

    # -- Velocity profile ----------------------------------------------
    vs     = res['vs']
    arc_ds = res['arc_ds']
    if vs:
        v_ln.set_data(range(len(vs)), vs)
        ax_v.set_xlim(0, len(vs))
        ax_v.set_ylim(0, max(params['v_max'] * 1.25, 0.1))

        # Mark constraint events: dashed vertical + triangle at speed limit
        for s_ev, v_ev in res['events']:
            ds = s_ev - res['s0']
            if ds < 0: continue
            arr_ds = np.asarray(arc_ds)
            idx    = int(np.searchsorted(arr_ds, ds))
            idx    = min(idx, len(vs) - 1)
            color  = 'crimson' if v_ev < 1e-3 else 'darkorange'
            ln     = ax_v.axvline(idx, color=color, alpha=0.35, lw=1.5, ls='--')
            mk,    = ax_v.plot(idx, v_ev, '^', color=color, ms=7, zorder=6)
            _event_arts.extend([ln, mk])

    # -- Corridor alpha --------------------------------------------------
    N = params['horizon']
    al_ln.set_data(range(N), [res['alpha']] * N)
    ax_al.set_xlim(0, N)

    # -- Status title ----------------------------------------------------
    n_segs = max(len(path) - 1, 1)
    ax_xy.set_title(
        f"Seg {res['seg']}/{n_segs - 1}   "
        f"CTE {res['cte']:.3f} m   "
        f"alpha {res['alpha']:.2f}   "
        f"v0 {_st['v_cur']:.2f} m/s",
        fontsize=10
    )

    fig.canvas.draw_idle()

# -- Main loop ---------------------------------------------------------

def loop():
    while plt.fignum_exists(fig.number):
        update()
        plt.pause(0.03)


loop()
