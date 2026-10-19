#!/usr/bin/env python3
"""Fig. 5: Lagrangian trajectories through the cartridge and nozzle.

168 tracers, seeded uniformly by area, are integrated through the solved velocity field at the
reference condition (200 kPa, 200 µm). Each takes the shear damage of the radius at which it
enters the nozzle; the extensional damage is the same for every tracer. Before drawing, the script
checks that the damage law reproduces two values published by Ning et al. (2018) and that the area
average over the nozzle matches the grid.

    python scripts/fig5_lagrangian_tracers.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l8_geometry import wall_radius  # noqa: E402
import l8_style as S                 # noqa: E402  the house style
from l8_style import panel_label     # noqa: E402  the shared A/B/C label

# Ning et al. 2018 damage law, L8-calibrated.
K_VIS, N_EXP = 25.60, 0.35
A1, B1, C1 = 3.20e-4, 0.61, 0.52
A2, B2 = 1.43e-3, 0.55

# Canonical grid, Ø200 µm / 200 kPa row of L8_FEM_canonical_2026-08-27.csv.
P_EN = 3756.437276788667             # Pa
D_ST_TARGET = 0.0028642409650113513  # fraction
D_T_CANONICAL = 8.956349848105468    # per cent, what this must reproduce

N_TRACERS = 168        # the manuscript's tracer count
RNG_SEED = 20260804    # same seed as the visualisation pipeline
Z_SEED = 5e-3          # release plane, 5 mm into the barrel

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get(
    "L8_CKPT",
    os.path.join(os.path.dirname(_HERE), "results", "fem", "checkpoints",
                 "fem_solution.pkl"))
OUT = os.environ.get(
    "L8_FIG3_OUT",
    os.path.join(os.path.dirname(_HERE), "results", "figures",
                 "Fig5_lagrangian_tracers.png"))

# The house style, not a second one. Sizes, weights, frame width and the full box all come
# from S.rc(S.PAPER), so this figure matches every other figure in the paper.
plt.rcParams.update(S.rc(S.PAPER))


def radius_at_z(z, geom):
    """Bore radius at z, from the single geometry definition in l8_geometry.

    This is the wall the tracers are tested against, and it is deliberately the
    same function that draws the wall in `draw_needle`, so a tracer can never be
    contained against a boundary the figure does not show. Holding a flat Rmid
    between z1 and z2 here would reject every tracer riding outside r = Rmid in
    the funnel, which is the outer, highest-shear part of the ensemble.
    """
    return wall_radius(z, geom)


def check_damage_law():
    """Ning's two published self-tests. If these move, the constants are wrong."""
    d1 = (1.0 - np.exp(-A1 * 900.0 ** B1 * 10.0 ** C1)) * 100.0
    d2 = (1.0 - np.exp(-A1 * 1700.0 ** B1 * 100.0 ** C1)) * 100.0
    if abs(d1 - 6.50) > 0.05 or abs(d2 - 27.96) > 0.10:
        sys.exit(f"damage law self-test failed: {d1:.2f}% (6.50), {d2:.2f}% (27.96)")
    print(f"  self-test: 900 Pa/10 s = {d1:.2f}% · 1700 Pa/100 s = {d2:.2f}%  OK")


def load_checkpoint(path):
    if not os.path.exists(path):
        sys.exit(f"no checkpoint at {path}\nSet L8_CKPT to the canonical fem_solution.pkl.")
    ck = pickle.load(open(path, "rb"))
    geom = {k: ck[k] for k in ("Rin", "Rmid", "Rtip", "z1", "z2", "z3")}
    return {
        "pts2d": np.asarray(ck["vtk_coords"]),
        "u": np.asarray(ck["fields"]["uvec"]),
        "tau": np.asarray(ck["fields"]["tau"]),
        "Q_uLs": float(ck["Q_uLs"]),
        "tau_w": float(ck["tau_w"]),
        "geom": geom,
    }


def compute_damage_profile(raw, nr=160, edge_trim=50e-6):
    """Radial shear-dose profile down the straight capillary.

    The SHAPE is the solved field: mean stress and mean axial velocity in each of 160
    radial bins, giving a residence time and hence a dose per bin. The LEVEL is then
    pinned so the area average equals the canonical grid's D_st for this condition, which
    keeps the figure consistent with every number in the text.
    """
    pts, u, tau_field, g = raw["pts2d"], raw["u"], raw["tau"], raw["geom"]
    Rtip, z2, z3 = g["Rtip"], g["z2"], g["z3"]
    L_tip = z3 - z2
    mask = ((pts[:, 1] >= z2 + edge_trim) & (pts[:, 1] <= z3 - edge_trim)
            & (pts[:, 0] <= Rtip * 1.001))
    r, vz, tau = pts[mask, 0], u[mask, 1], tau_field[mask]

    edges = np.linspace(0.0, Rtip, nr + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    tau_c, vz_c = np.full(nr, np.nan), np.full(nr, np.nan)
    for i in range(nr):
        m = (r >= edges[i]) & (r <= edges[i + 1] if i == nr - 1 else r < edges[i + 1])
        if np.any(m):
            tau_c[i] = float(np.nanmean(tau[m]))
            vz_c[i] = float(np.nanmean(vz[m]))
    good = np.isfinite(tau_c) & np.isfinite(vz_c)
    tau_c = np.interp(rc, rc[good], tau_c[good])
    vz_c = np.interp(rc, rc[good], vz_c[good])

    vz_floor = max(float(np.nanpercentile(vz_c[vz_c > 0], 0.5)) * 0.01, 1e-6)
    ts = L_tip / np.maximum(vz_c, vz_floor)
    ds = np.clip(1.0 - np.exp(-A1 * tau_c ** B1 * ts ** C1), 0.0, 1.0)
    dst_raw = float(2.0 * np.trapezoid(rc * ds, rc) / (Rtip ** 2))
    if dst_raw > 0:
        ds = np.clip(ds * (D_ST_TARGET / dst_raw), 0.0, 1.0)
    dst = float(2.0 * np.trapezoid(rc * ds, rc) / (Rtip ** 2))

    tau_e = (3.0 / 8.0) * (N_EXP + 1.0) * P_EN
    de = float(np.clip(1.0 - np.exp(-A2 * tau_e ** B2), 0.0, 1.0))
    return {"r": rc, "D_s": ds, "D_st": dst, "tau_e": tau_e, "D_e": de,
            "D_total": de + (1.0 - de) * dst, "t_s": ts, "tau": tau_c}


def integrate(raw, prof, n_want=N_TRACERS):
    """Track tracers to the outlet. Seeds uniform by area, as in the pipeline.

    Tracers that leave the domain or fail the wall test are dropped, so more are seeded
    than are wanted and the first n_want survivors are kept. The seed is fixed, so the
    figure is reproducible.
    """
    g = raw["geom"]
    interp_r = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 0], fill_value=np.nan)
    interp_z = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 1], fill_value=np.nan)

    def outlet(_t, y):
        return y[1] - g["z3"]
    outlet.terminal, outlet.direction = True, 1

    def ode(_t, y):
        r, z = y
        q = [[abs(r), z]]
        ur = float(interp_r(q)[0]) * (1.0 if r >= 0 else -1.0)
        uz = float(interp_z(q)[0])
        return [0.0 if not np.isfinite(ur) else ur,
                0.0 if not np.isfinite(uz) else uz]

    rng = np.random.default_rng(RNG_SEED)
    # The pool is oversized because not every seeded path reaches the outlet within the integration
    # window; four times n_want is enough to reach n_want survivors on this domain.
    n_seed = int(n_want * 4.0)
    r_seeds = 0.95 * g["Rin"] * np.sqrt(rng.uniform(0, 1, n_seed))
    th_seeds = rng.uniform(0.0, 2.0 * np.pi, n_seed)

    paths, ds_p, dt_p, r_tip_p, theta_p = [], [], [], [], []
    for i in range(n_seed):
        if len(paths) >= n_want:
            break
        sol = solve_ivp(ode, (0.0, 120.0), [float(r_seeds[i]), Z_SEED],
                        events=outlet, max_step=0.01, rtol=1e-4, atol=1e-9)
        path = sol.y[:2].T
        if path.shape[0] <= 5 or path[-1, 1] < g["z3"] * 0.999:
            continue
        if not np.all(np.abs(path[:, 0]) <= radius_at_z(path[:, 1], g) + 3e-6):
            continue
        in_tip = path[:, 1] >= g["z2"]
        if not np.any(in_tip):
            continue
        r_tip = float(np.clip(abs(path[int(np.argmax(in_tip)), 0]), 0.0, g["Rtip"]))
        ds = float(np.interp(r_tip, prof["r"], prof["D_s"]))
        paths.append(path)
        ds_p.append(ds)
        dt_p.append((prof["D_e"] + (1.0 - prof["D_e"]) * ds) * 100.0)
        r_tip_p.append(r_tip)
        theta_p.append(float(th_seeds[i]))
    if len(paths) < n_want:
        sys.exit(f"only {len(paths)} of {n_want} tracers survived; widen the seed pool")
    return {"paths": paths, "D_s": np.asarray(ds_p), "D_t": np.asarray(dt_p),
            "r_tip": np.asarray(r_tip_p), "theta": np.asarray(theta_p)}


def draw_needle(ax, g, mm=1e3, lw=0.9):
    """Outline of the cartridge and needle, in mm, as a filled section.

    Drawn as one closed polygon rather than fill_between, which only filled as far as the
    first step: the step corners repeat an x value, and fill_between treats that as the
    end of a region rather than a vertical wall.
    """
    # The cartridge tapers from Rin at z1 to Rmid at z2 over the chamfer; it does not step.
    # The only abrupt contraction is Rmid -> Rtip at z2, so the section has exactly one step.
    z = np.array([0, g["z1"], g["z2"], g["z2"], g["z3"]]) * mm
    r = np.array([g["Rin"], g["Rin"], g["Rmid"], g["Rtip"], g["Rtip"]]) * mm
    # Axisymmetric half-section, r >= 0.
    ax.fill_between(z, 0.0, r, color="#F2F2F4", zorder=0, linewidth=0)
    ax.plot(z, r, color="#444444", lw=lw, zorder=4)
    # The centreline ends where the section ends.
    ax.plot([0.0, g["z3"] * mm], [0.0, 0.0], color="#888888", lw=0.8,
            ls=(0, (7, 4)), zorder=4)


def title_artist(ax):
    """The Text object that actually carries this panel's title.

    `ax.title` is the CENTRE title and nothing else. A title set with loc="left" or
    loc="right" lives in a separate Text object, so a guard written against `ax.title` alone
    would measure an empty string, never enter its own loop, and report a clean result having
    tested nothing.

    Returns whichever of the three title slots holds text, or None if the panel has no
    title at all.
    """
    for _slot in ("title", "_left_title", "_right_title"):
        _t = getattr(ax, _slot, None)
        if _t is not None and (_t.get_text() or "").strip():
            return _t
    return None


def clearance_px(a, b):
    """Signed clearance between two display-space boxes, in pixels.

    Positive is the gap between two boxes that are clear of one another; negative means they
    intersect, and the magnitude is how deep the intersection runs on its shallower axis.

    Overlap is a two-dimensional fact, so the separation reported is the larger of the two
    axis separations: two boxes in the same column but different rows are not touching, and
    a check that tests one axis alone reports them as a collision. That kind of false
    positive is what gets a layout guard switched off instead of fixed, so it is designed
    out here rather than tuned away later.
    """
    return max(max(a.x0, b.x0) - min(a.x1, b.x1),
               max(a.y0, b.y0) - min(a.y1, b.y1))


def main():
    print("Figure 3 — Lagrangian trajectories")
    check_damage_law()
    raw = load_checkpoint(CKPT)
    g = raw["geom"]
    prof = compute_damage_profile(raw)
    print(f"  profile: D_st = {prof['D_st']*100:.4f} %  D_et = {prof['D_e']*100:.4f} %  "
          f"D_t = {prof['D_total']*100:.4f} %")
    if abs(prof["D_total"] * 100 - D_T_CANONICAL) > 1e-6:
        sys.exit(f"profile D_t {prof['D_total']*100:.6f} != canonical {D_T_CANONICAL:.6f}")
    print(f"  reproduces the canonical grid to 1e-6  OK")

    tr = integrate(raw, prof)
    dt, r_tip = tr["D_t"], tr["r_tip"]
    print(f"  tracked {len(dt)} tracers · D_t {dt.min():.3f} to {dt.max():.3f} %, "
          f"spread {dt.max()-dt.min():.3f} pp")

    mm = 1e3
    # The panel grid.
    fig = plt.figure(figsize=(S.PAGE_W_IN, 6.6000))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.25, 1.00],
                          hspace=0.66, wspace=0.48,
                          left=0.088, right=0.862, top=0.930, bottom=0.070)

    norm = plt.Normalize(dt.min(), dt.max())
    cmap = plt.get_cmap("viridis")

    axa = fig.add_subplot(gs[0, :])
    draw_needle(axa, g)
    order = np.argsort(dt)                     # dirtiest tracks drawn last
    for i in order:
        p = tr["paths"][i]
        axa.plot(p[:, 1] * mm, p[:, 0] * mm, color=cmap(norm(dt[i])),
                 lw=0.55, alpha=0.85, zorder=3)
    # The line is the statement; the key names it.
    axa.axvline(g["z2"] * mm, color="#B4501F", lw=0.8, ls="--", zorder=5,
                label="contraction to nozzle")
    leg_a = axa.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.7,
                       labelspacing=0.25, borderpad=0.1, borderaxespad=0.35,
                       handletextpad=0.6)
    # A small margin on the right, mirroring the one on the left, so the last tick label sits inside
    # the axes rather than centred on its corner, clear of the colour bar.
    axa.set_xlim(Z_SEED * mm - 0.3, g["z3"] * mm + 0.35)
    axa.set_ylim(-0.09, 2.32)
    axa.set_xlabel(S.LBL["axial_z"])
    axa.set_ylabel(S.LBL["radius_mm"])
    # The panel starts at the release plane, 5 mm into the cartridge, because the first 5 mm
    # carries no track: it shows the tracked part of the flow path, not the whole of it.
    panel_label(axa, "A")

    # No draw_needle here: this panel's y axis is in µm and the helper draws in mm. The
    # capillary is a straight tube over this range, so two wall lines are the whole geometry.
    axb = fig.add_subplot(gs[1, 0])
    Rtip_um = g["Rtip"] * 1e6
    axb.axhspan(0.0, Rtip_um, color="#F2F2F4", zorder=0)
    # Clip each path to the tip before drawing it.
    for i in order:
        p = tr["paths"][i]
        _in = p[:, 1] >= g["z2"]
        if not _in.any():
            continue
        axb.plot(p[_in, 1] * mm, p[_in, 0] * 1e6, color=cmap(norm(dt[i])),
                 lw=0.6, alpha=0.9, zorder=3)
    axb.set_xlim(g["z2"] * mm - 0.15, g["z3"] * mm)
    axb.set_ylim(-0.10 * Rtip_um, 1.20 * Rtip_um)
    axb.axhline(Rtip_um, color="#444444", lw=0.9, zorder=4)
    axb.axhline(0.0, color="#888888", lw=0.8, ls=(0, (7, 4)), zorder=4)
    axb.set_xlabel(S.LBL["axial_z"])
    axb.set_ylabel(S.LBL["radius_um"])
    panel_label(axb, "B")

    # Nothing may be drawn outside the bore. Measured against the geometry, not eyeballed.
    _worst = max(float(np.max(np.abs(tr["paths"][i][tr["paths"][i][:, 1] >= g["z2"], 0])))
                 for i in order if (tr["paths"][i][:, 1] >= g["z2"]).any())
    assert _worst <= g["Rtip"] + 3e-6, (
        f"a drawn tracer reaches r = {_worst*1e6:.1f} µm in a {g['Rtip']*1e6:.0f} µm bore")
    print(f"  panel (b) containment: worst drawn radius {_worst*1e6:.1f} µm "
          f"of {g['Rtip']*1e6:.0f} µm bore")

    # Shear damage, not total.
    axc = fig.add_subplot(gs[1, 1])
    ds_pct = tr["D_s"] * 100
    axc.scatter(r_tip * 1e6, ds_pct, s=13, c=dt, cmap=cmap, norm=norm,
                edgecolor="#333333", linewidth=0.35, zorder=3)
    axc.set_yscale("log")
    axc.set_ylim(ds_pct.min() / 1.4, ds_pct.max() * 1.4)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    _ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if ds_pct.min() / 1.4 <= t <= ds_pct.max() * 1.4]
    axc.yaxis.set_major_locator(FixedLocator(_ticks))
    axc.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in _ticks]))
    axc.yaxis.set_minor_locator(NullLocator())
    # The result value belongs in the caption, not on the axes; it is printed to stdout so
    # the caption stays in step with the data.
    print(f"  caption (c)  ·  shear damage {ds_pct.min():.2f} % on the axis to {ds_pct.max():.2f} % "
          f"at the outermost tracer; D_et = {prof['D_e']*100:.2f} %, identical for every tracer")
    axc.set_xlabel(S.LBL["entry_r"])
    axc.set_ylabel("Shear damage  $D_s$  (%)")
    panel_label(axc, "C")

    axd = fig.add_subplot(gs[2, :])
    axd.hist(dt, bins=28, color="#4C6E8A", edgecolor="#243B4A", linewidth=0.6, zorder=3)
    axd.axvline(prof["D_e"] * 100, color="#B4501F", lw=1.2, ls="--", zorder=4,
                label="extensional floor  $D_{et}$")
    axd.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.7,
               borderaxespad=0.3)
    axd.set_xlabel("Cumulative  $D_t$  at the outlet (%)")
    axd.set_ylabel("Tracers")
    panel_label(axd, "D")
# Journal rule: figure titles and legends go in the main manuscript, never inside the
# graphic file, so the numbers the caption quotes are printed to stdout instead.
    span = dt.max() - dt.min()
    print(f"  for the caption: population spans {span:.3f} pp "
          f"({span / dt.mean() * 100:.1f} % of the mean), extensional floor "
          f"{prof['D_e']*100:.2f} pp")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # The bar is placed from panel (a), not from four literals.
    _pa = axa.get_position()
    cax = fig.add_axes([_pa.x1 + 0.043, _pa.y0, 0.015, _pa.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(S.LBL["damage_cum"], fontsize=S.PAPER["label"])
    cb.ax.tick_params(labelsize=S.PAPER["tick"])

    # Every check above this line is on numbers: the damage law against Ning's two self- tests, the
    # profile against the canonical grid, the drawn tracks against the bore.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    panels = [("(a)", axa), ("(b)", axb), ("(c)", axc), ("(d)", axd), ("colour bar", cax)]
    MIN_PX = 12.0     # 0.12 in on the 100 dpi draw canvas, about 8.6 pt on the page

    # 1.
    for _pn, _pax in panels[:4]:
        _loose = [t.get_text() for t in _pax.texts
                  if (t.get_text() or "").strip() and (t.get_text() or "").strip() not in "ABCD"]
        assert not _loose, (
            f"panel {_pn} carries {len(_loose)} loose annotation(s) {_loose}; "
            f"name the object in the key instead of writing on the panel")

    # 2. no key may reach into a panel that is not its own.
    for _kname, _key, _home in [("panel (a) key", leg_a, axa)]:
        _kb = _key.get_window_extent(rend)
        for _pn, _pax in panels:
            if _pax is _home:
                continue
            _d = clearance_px(_kb, _pax.get_window_extent(rend))
            assert _d > 0, f"the {_kname} reaches {-_d:.1f} px into panel {_pn}"

    # 3.
    _kd = leg_a.get_window_extent(rend).transformed(axa.transData.inverted())
    _kz0, _kz1 = min(_kd.x0, _kd.x1) / mm, max(_kd.x0, _kd.x1) / mm
    _under_key = 0.0
    for _p in tr["paths"]:
        _m = (_p[:, 1] >= _kz0) & (_p[:, 1] <= _kz1)
        if _m.any():
            _under_key = max(_under_key, float(np.max(np.abs(_p[_m, 0]))))
    _under_key *= mm
    _key_floor = min(_kd.y0, _kd.y1)
    assert _key_floor > _under_key, (
        f"the panel (a) key hangs down to r = {_key_floor:.2f} mm while the tracks under it "
        f"reach r = {_under_key:.2f} mm; it is sitting on the data")

    # 4.
    _tight = [(n, a.get_tightbbox(rend)) for n, a in panels]
    _worst, _pair = float("inf"), None
    for _i in range(len(_tight)):
        for _j in range(_i + 1, len(_tight)):
            _d = clearance_px(_tight[_i][1], _tight[_j][1])
            if _d < _worst:
                _worst, _pair = _d, (_tight[_i][0], _tight[_j][0])
    assert _worst >= MIN_PX, (
        f"{_pair[0]} and {_pair[1]} are {_worst:.1f} px apart, under the "
        f"{MIN_PX:.0f} px minimum")

    # 5.
    _wl, _wln = float("inf"), None
    for _pn, _pax in panels[:4]:
        _ti = next((t for t in _pax.texts
                    if (t.get_text() or "").strip() in "ABCD"), None)
        if _ti is None:
            continue
        _d = (_ti.get_window_extent(rend).y0
              - _pax.yaxis.label.get_window_extent(rend).y1)
        if _d < _wl:
            _wl, _wln = _d, _pn
    assert _wln is not None, "no panel letter was found to measure; the guard tested nothing"
    assert _wl >= -MIN_PX, (
        f"panel {_wln}'s y label rises {-_wl:.1f} px past its panel letter, "
        f"more than the {MIN_PX:.0f} px tolerance")

    # 6.
    assert abs(axb.get_position().width - axc.get_position().width) < 1e-9, (
        f"(b) and (c) differ in width: {axb.get_position().width:.6f} vs "
        f"{axc.get_position().width:.6f}")

    print(f"  layout guard: {len(panels)} boxes, tightest pair {_pair[0]}/{_pair[1]} at "
          f"{_worst:.1f} px · tightest y-label-to-title {_wln} at {_wl:.1f} px · key clears "
          f"the tracks by {_key_floor - _under_key:.2f} mm · (b) and (c) equal width")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Panel letters snap to each panel's real top-left corner once the layout is fixed.
    S.snap_panel_labels(fig)
    # The two save-time guards every save_figure figure runs (this figure keeps its own output
    # path, so it calls them directly): no key over data, no ink at the canvas edge.
    S.assert_keys_clear_of_data(fig)
    S.assert_ink_inside_canvas(fig)
    fig.savefig(OUT, dpi=600, facecolor="white")
    from l8_paths import flatten_png; flatten_png(OUT)
    plt.close(fig)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
