#!/usr/bin/env python3
"""Fig. 5: Lagrangian trajectories through the cartridge and nozzle.

Tracers are released over the inner 95 % of the cartridge radius on a plane 5 mm into the cartridge
and integrated through the solved velocity field at the reference condition (200 kPa, 200 µm). Each
takes the shear damage Ds of Eq. 20 at the radius at which it travels through the nozzle, the median
radius of its path beyond the first 1 mm, where the paths have stopped converging; the entrance damage
is the same for every tracer.

(A) The 168 tracks from the release plane to the outlet and (B) inside the nozzle, coloured by the
condition-dependent damage Dn. (C) Ds, on a logarithmic axis, against the radius in the nozzle.
(D) Distribution of Dn over the tracers, with the entrance-associated damage common to every tracer
dashed. How the tracer statistics depend on the tracer set is tracking_sensitivity.py (Additional
file 1, Table S2).

Before drawing, the script checks that the damage law reproduces two values published by Ning et al.
(2018), that the profile reproduces the Dn of the grid and that the tracks it draws reproduce the
per-tracer table of tracking_sensitivity.py. The tracks are read from
results/analyses/tracking_published_paths_cache.npz, which holds the paths of that table; without it they are
integrated again and must match the table to 0.5 µm in radius and 0.01 percentage points in Ds. The numbers
of the caption go to results/analyses/fig5_caption_numbers.json.

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
# ODE tolerances of every tracer path.
RTOL, ATOL = 1e-8, 1e-13

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


def nozzle_radius(path, g, skip=1e-3):
    """The radius a tracer travels at through the straight nozzle: the median |r| of its path beyond the first
    `skip` of the nozzle. At the inlet plane the paths are still converging (the outermost settles about 3 µm
    inward within 0.1 mm of the 10 mm nozzle), so the radius at entry is not the radius the shear acts at."""
    m = (path[:, 1] >= g["z2"] + skip) & (path[:, 1] <= g["z3"])
    return float(min(np.median(np.abs(path[m, 0])), g["Rtip"]))


def entry_radius(path, g):
    """|r| where a tracer crosses the nozzle inlet plane z = z2: a cubic in z through the two stored path points on
    either side of the plane. The first stored point past the plane lies an arbitrary distance beyond it, set by
    where the adaptive steps fall, and the radius read there can sit about 1 µm inside the crossing radius."""
    j = int(np.argmax(path[:, 1] >= g["z2"]))
    k = slice(max(j - 2, 0), min(j + 2, len(path)))
    zz, rr = path[k, 1] - g["z2"], path[k, 0]
    return float(min(abs(np.polyval(np.polyfit(zz, rr, min(3, len(zz) - 1)), 0.0)), g["Rtip"]))


def shear_damage_function(g):
    """D_s(r) of Eq. 20 at the reference condition: the integrand of the area average of Eq. 21, so the tracers and
    Eq. 21 use one law. Checked against the canonical D_st before use."""
    from l8_damage import REFERENCE, compute_Dst, derive, shear_damage_at
    dp = derive(REFERENCE["P_kPa"], REFERENCE["Q_uLs"], g["Rtip"])["dP_tip_kPa"] * 1e3
    assert abs(compute_Dst(dp, g["Rtip"]) - D_ST_TARGET) < 1e-9, "Eq. 21 does not reproduce the canonical D_st"
    return lambda r: shear_damage_at(r, dp, g["Rtip"])


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
                        events=outlet, max_step=0.01, rtol=RTOL, atol=ATOL)
        path = sol.y[:2].T
        if path.shape[0] <= 5 or path[-1, 1] < g["z3"] * 0.999:
            continue
        if not np.all(np.abs(path[:, 0]) <= radius_at_z(path[:, 1], g) + 3e-6):
            continue
        in_tip = path[:, 1] >= g["z2"]
        if not np.any(in_tip):
            continue
        r_tip = nozzle_radius(path, g)
        ds = float(shear_damage_function(g)(r_tip))
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


def load_tracks(raw, prof):
    """The published 168 tracks: from the cache of tracking_sensitivity.py when it was written
    from this checkpoint, otherwise tracked here. Either way the per-tracer quantities are derived from
    the paths by the same rule as `integrate`."""
    import hashlib
    from l8_paths import DATA
    g = raw["geom"]
    cache = DATA / "analyses" / "tracking_published_paths_cache.npz"
    md5 = hashlib.md5(open(CKPT, "rb").read()).hexdigest()
    z = np.load(cache) if cache.exists() else None
    if z is not None and str(z["ckpt_md5"]) == md5 and "rtol" in z.files and float(z["rtol"]) == RTOL \
            and float(z["atol"]) == ATOL:
        paths = [z["xy"][a:b] for a, b in zip(z["ptr"][:-1], z["ptr"][1:])]
        cached = True
        print(f"  tracks: {len(paths)} read from {cache.name} (checkpoint md5 and tolerances match)")
    else:
        paths = integrate(raw, prof)["paths"]
        cached = False
        print(f"  tracks: {len(paths)} integrated here")
    if len(paths) != N_TRACERS:
        sys.exit(f"{len(paths)} tracks, expected {N_TRACERS}")
    ds_at = shear_damage_function(g)
    ds_p, r_tip_p = [], []
    for path in paths:
        r_noz = nozzle_radius(path, g)
        r_tip_p.append(r_noz)
        ds_p.append(float(ds_at(r_noz)))
    ds_p = np.asarray(ds_p)
    return {"paths": paths, "D_s": ds_p, "D_t": (prof["D_e"] + (1.0 - prof["D_e"]) * ds_p) * 100.0,
            "r_tip": np.asarray(r_tip_p), "cached": cached}


SETS = (("published", "{kept} tracers\n$r_0 \\leq 0.95\\,R_\\mathrm{{in}}$"),
        ("pool_0.95", "{kept} tracers\n$r_0 \\leq 0.95\\,R_\\mathrm{{in}}$"),
        ("pool_0.99", "{kept} of {seeded}\n$r_0 \\leq 0.99\\,R_\\mathrm{{in}}$"),
        ("flux_0.99", "{kept} of {seeded}\nflow-weighted\n$r_0 \\leq 0.99\\,R_\\mathrm{{in}}$"))


def load_sets():
    """Per-tracer shear damage of the four sets, and the summary they must reproduce."""
    import csv
    import json
    from l8_paths import DATA
    rows = list(csv.DictReader(open(DATA / "analyses" / "tracking_tracers.csv")))
    summ = json.loads((DATA / "analyses" / "tracking_sensitivity.json").read_text())
    out = {}
    for (sid, _), s in zip(SETS, summ["sets"]):
        ds = np.array([float(r["D_s_pct"]) for r in rows if r["set"] == sid])
        er = np.array([float(r["nozzle_r_um"]) for r in rows if r["set"] == sid])
        assert len(ds) == s["kept"], (sid, len(ds), s["kept"])
        assert abs(ds.mean() - s["mean_Dst_pct"]) < 5e-5 and abs(np.percentile(ds, 95) - s["p95_Dst_pct"]) < 5e-5, sid
        out[sid] = {"D_s": ds, "nozzle_r_um": er, "summary": s}
    return out, summ


def main():
    print("Figure 5 — Lagrangian trajectories")
    check_damage_law()
    raw = load_checkpoint(CKPT)
    g = raw["geom"]
    prof = compute_damage_profile(raw)
    print(f"  profile: D_st = {prof['D_st']*100:.4f} %  D_et = {prof['D_e']*100:.4f} %  "
          f"D_n = {prof['D_total']*100:.4f} %")
    if abs(prof["D_total"] * 100 - D_T_CANONICAL) > 1e-6:
        sys.exit(f"profile D_n {prof['D_total']*100:.6f} != canonical {D_T_CANONICAL:.6f}")
    print("  reproduces the canonical grid to 1e-6  OK")

    tr = load_tracks(raw, prof)
    sets, summ = load_sets()
    # The cached tracks are the table's tracks: same radius in the nozzle, same D_s, tracer by tracer.
    pub = sets["published"]
    tol_r, tol_ds = (1e-5, 1e-6) if tr["cached"] else (0.05, 0.002)
    dr = np.abs(np.sort(tr["r_tip"] * 1e6) - np.sort(pub["nozzle_r_um"])).max()
    dds = np.abs(np.sort(tr["D_s"] * 100) - np.sort(pub["D_s"])).max()
    assert dr <= tol_r, f"tracks differ from the table by {dr:.3g} µm"
    assert dds <= tol_ds, f"D_s differs from the table by {dds:.3g} pp"
    print(f"  tracks against the table: radii within {dr:.2g} µm, D_s within {dds:.2g} pp")
    dt, r_tip = tr["D_t"], tr["r_tip"]
    ds_pct = tr["D_s"] * 100
    r_star = float(r_tip.max())          # the outermost radius at which the published set travels the nozzle
    print(f"  tracked {len(dt)} tracers · D_n {dt.min():.3f} to {dt.max():.3f} %, spread {dt.max()-dt.min():.3f} pp")

    # Reference averages, and the share of the flow in the stream the 168 tracers do not sample.
    from l8_damage import derive, REFERENCE
    ref = derive(REFERENCE["P_kPa"], REFERENCE["Q_uLs"], g["Rtip"])
    r0_star = max(abs(float(p[0, 0])) for p in tr["paths"])
    iz = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 1], fill_value=np.nan)
    rr = np.linspace(0.0, g["Rin"], 20001)
    uz = np.nan_to_num(iz(np.column_stack([rr, np.full_like(rr, Z_SEED)])), nan=0.0)
    flux = np.cumsum(np.r_[0.0, 0.5 * (rr[1:] * uz[1:] + rr[:-1] * uz[:-1]) * np.diff(rr)])
    q_plane = 2 * np.pi * float(flux[-1]) * 1e9
    assert abs(q_plane / raw["Q_uLs"] - 1) < 0.02, f"release-plane flow {q_plane:.3f} against {raw['Q_uLs']:.3f} µL/s"
    layer_flow = 1.0 - float(np.interp(r0_star, rr, flux)) / float(flux[-1])
    print(f"  release plane: flow {q_plane:.3f} µL/s against {raw['Q_uLs']:.3f} solved; outermost release "
          f"{r0_star*1e3:.4f} mm = {r0_star/g['Rin']:.4f} R_in; outside it {layer_flow*100:.2f} % of the flow")
    # The same share from the developed nozzle profile outside the radius the outermost tracer travels at: the two
    # agree if the tracked stream tube is the developed one, which is what reading D_s at that radius assumes.
    from l8_damage import velocity_profile
    ds_fn = shear_damage_function(g)
    dp = ref["dP_tip_kPa"] * 1e3
    rn = np.linspace(0.0, g["Rtip"], 200001)
    vn = velocity_profile(dp, g["Rtip"], rn)
    fn = np.cumsum(np.r_[0.0, 0.5 * (rn[1:] * vn[1:] + rn[:-1] * vn[:-1]) * np.diff(rn)])
    layer_flow_noz = 1.0 - float(np.interp(r_star, rn, fn)) / float(fn[-1])
    # Near the wall the flow outside r grows as (R - r)^2, so the two shares are compared as radii: the nozzle radius
    # whose annulus carries the release-plane share must lie within the ~1 µm scatter of the tracked radii.
    r_equiv = float(np.interp(1.0 - layer_flow, fn / fn[-1], rn))
    assert abs(r_equiv - r_star) < 1e-6, (r_equiv, r_star)
    print(f"  nozzle: outside {r_star*1e6:.2f} µm the developed profile carries {layer_flow_noz*100:.2f} % of the flow; "
          f"the release-plane share {layer_flow*100:.2f} % corresponds to {r_equiv*1e6:.2f} µm "
          f"({(r_star - r_equiv)*1e6:.2f} µm apart)")

    def t_res(r):
        return (g["z3"] - g["z2"]) / float(velocity_profile(dp, g["Rtip"], r))
    d_area, d_flow = ref["Dst_pct"], ref["Dst_flux_pct"]
    assert abs(d_area - prof["D_st"] * 100) < 1e-6, (d_area, prof["D_st"])
    p99 = sets["pool_0.99"]["D_s"]
    pfx = sets["flux_0.99"]["D_s"]
    print(f"  caption: D_s {ds_pct.min():.2f} % for the innermost tracer to {ds_pct.max():.2f} % for the outermost "
          f"(radius in the nozzle {r_star*1e6:.2f} um = {r_star/g['Rtip']*100:.1f} % of R, {(g['Rtip']-r_star)*1e6:.2f} um from the wall); "
          f"D_et = {prof['D_e']*100:.2f} %")
    print(f"  caption: area average {d_area:.3f} %, flow-weighted average {d_flow:.3f} %")
    print(f"  caption: tracers above {ds_pct.max():.3f} % shear damage: {np.mean(p99 > ds_pct.max())*100:.1f} % of the "
          f"0.99 set, {np.mean(pfx > ds_pct.max())*100:.1f} % of the flow-weighted set")
    for sid, lab in SETS:
        s = sets[sid]["summary"]
        print(f"  set {sid:10s} kept {s['kept']:4d} of {s['seeded']:4d}: mean {s['mean_Dst_pct']:.3f} %, "
              f"p95 {s['p95_Dst_pct']:.3f} %, max {s['max_Dst_pct']:.3f} %")

    # The caption's numbers, written to results/analyses/fig5_caption_numbers.json.
    import json
    from l8_paths import DATA
    cap = dict(n_tracers=len(dt), Ds_min_pct=round(float(ds_pct.min()), 4), Ds_max_pct=round(float(ds_pct.max()), 4),
               Det_pct=round(prof["D_e"] * 100, 4), Dn_spread_pp=round(float(dt.max() - dt.min()), 4),
               nozzle_r_max_um=round(r_star * 1e6, 3), layer_um=round((g["Rtip"] - r_star) * 1e6, 3),
               entry_r_max_um=round(max(entry_radius(p_, g) for p_ in tr["paths"]) * 1e6, 3),
               release_r_max_mm=round(r0_star * 1e3, 4), layer_flow_release_pct=round(layer_flow * 100, 3),
               layer_flow_pct=round(layer_flow_noz * 100, 3), layer_r_offset_um=round((r_star - r_equiv) * 1e6, 2),
               area_average_pct=round(d_area, 4),
               flow_weighted_pct=round(d_flow, 4), n_tail_099=int(np.sum(sets["pool_0.99"]["nozzle_r_um"] > r_star * 1e6)),
               frac_above_168max_099_pct=round(float(np.mean(p99 > ds_pct.max())) * 100, 2),
               frac_above_168max_flux_pct=round(float(np.mean(pfx > ds_pct.max())) * 100, 2),
               r_min_um=round(float(r_tip.min()) * 1e6, 2), t_res_inner_ms=round(t_res(r_tip.min()) * 1e3, 1),
               t_res_outer_ms=round(t_res(r_star) * 1e3, 1), Dn_mean_pct=round(float(dt.mean()), 3),
)
    (DATA / "analyses" / "fig5_caption_numbers.json").write_text(json.dumps(cap, indent=1))
    print(f"  caption numbers: {cap}")

    mm = 1e3
    # The submitted layout.
    fig = plt.figure(figsize=(S.PAGE_W_IN, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.25, 1.00], hspace=0.66, wspace=0.48,
                          left=0.088, right=0.862, top=0.930, bottom=0.070)
    norm = plt.Normalize(dt.min(), dt.max())
    cmap = plt.get_cmap(S.CMAP_DT)
    order = np.argsort(dt)                     # most damaged tracks drawn last

    axa = fig.add_subplot(gs[0, :])
    draw_needle(axa, g)
    for i in order:
        p = tr["paths"][i]
        axa.plot(p[:, 1] * mm, p[:, 0] * mm, color=cmap(norm(dt[i])), lw=0.55, alpha=0.85, zorder=3)
    # The contraction is named in a key, not by text laid on the tracks.
    axa.axvline(g["z2"] * mm, color=S.ANNOT, lw=0.8, ls="--", zorder=5, label="contraction to nozzle")
    leg_a = axa.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.7, labelspacing=0.25,
                       borderpad=0.1, borderaxespad=0.35, handletextpad=0.6)
    axa.set_xlim(Z_SEED * mm - 0.3, g["z3"] * mm + 0.35)
    axa.set_ylim(-0.09, 2.32)
    axa.set_xlabel(S.LBL["axial_z"])
    axa.set_ylabel(S.LBL["radius_mm"])
    panel_label(axa, "A")

    # Each path is clipped to the nozzle before drawing: consecutive samples straddle the step at z2, and the
    # segment joining them would be drawn above the 100 µm wall.
    axb = fig.add_subplot(gs[1, 0])
    Rtip_um = g["Rtip"] * 1e6
    axb.axhspan(0.0, Rtip_um, color=S.PANEL, zorder=0)
    for i in order:
        p = tr["paths"][i]
        _in = p[:, 1] >= g["z2"]
        if _in.any():
            axb.plot(p[_in, 1] * mm, p[_in, 0] * 1e6, color=cmap(norm(dt[i])), lw=0.6, alpha=0.9, zorder=3)
    axb.set_xlim(g["z2"] * mm - 0.15, g["z3"] * mm)
    axb.set_ylim(-0.10 * Rtip_um, 1.20 * Rtip_um)
    axb.axhline(Rtip_um, color="#444444", lw=0.9, zorder=4)
    axb.axhline(0.0, color="#888888", lw=0.8, ls=(0, (7, 4)), zorder=4)
    axb.set_xlabel(S.LBL["axial_z"])
    axb.set_ylabel(S.LBL["radius_um"])
    panel_label(axb, "B")
    _worst = max(float(np.max(np.abs(p[p[:, 1] >= g["z2"], 0]))) for p in tr["paths"] if (p[:, 1] >= g["z2"]).any())
    assert _worst <= g["Rtip"] + 3e-6, f"a drawn tracer reaches r = {_worst*1e6:.1f} µm in a {Rtip_um:.0f} µm bore"
    print(f"  panel B containment: worst drawn radius {_worst*1e6:.1f} µm of {Rtip_um:.0f} µm bore")

    # Shear damage, not D_n: on D_n the tenfold rise sits on the common entrance term and reads as a wiggle.
    axc = fig.add_subplot(gs[1, 1])
    axc.scatter(r_tip * 1e6, ds_pct, s=13, c=dt, cmap=cmap, norm=norm, edgecolor=S.INK, linewidth=0.35, zorder=3)
    axc.set_yscale("log")
    lo_c, hi_c = ds_pct.min() / 1.4, ds_pct.max() * 1.4
    axc.set_ylim(lo_c, hi_c)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    _ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if lo_c <= t <= hi_c]
    axc.yaxis.set_major_locator(FixedLocator(_ticks))
    axc.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in _ticks]))
    axc.yaxis.set_minor_locator(NullLocator())
    axc.set_xlabel("Radius in the nozzle (µm)")
    axc.set_ylabel("Shear damage  $D_s$  (%)")
    panel_label(axc, "C")

    axd = fig.add_subplot(gs[2, :])
    axd.hist(dt, bins=28, color="#4C6E8A", edgecolor="#243B4A", linewidth=0.6, zorder=3)
    axd.axvline(prof["D_e"] * 100, color=S.ANNOT, lw=1.2, ls="--", zorder=4,
                label="entrance-associated damage  $D_{et}$")
    leg_d = axd.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.7, borderaxespad=0.3)
    axd.set_xlabel("Damage  $D_n$  at the outlet (%)")
    axd.set_ylabel("Tracers")
    panel_label(axd, "D")

    # The colour bar is placed from panel A, so it is that panel's height and reads as its scale.
    _pa = axa.get_position()
    cax = fig.add_axes([_pa.x1 + 0.043, _pa.y0, 0.015, _pa.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_label("Damage  $D_n$  (%)", fontsize=S.PAPER["label"])
    cb.ax.tick_params(labelsize=S.PAPER["tick"])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    panels = [("A", axa), ("B", axb), ("C", axc), ("D", axd), ("colour bar", cax)]
    MIN_PX = 12.0
    # 1. No loose text on a data panel; the panel letter is the only text.
    for _pn, _pax in panels[:4]:
        _loose = [t.get_text() for t in _pax.texts if (t.get_text() or "").strip() not in "ABCD"]
        assert not _loose, f"panel {_pn} carries loose text {_loose}"
    # 2. No key reaches into another panel.
    for _kname, _key, _home in (("A key", leg_a, axa), ("D key", leg_d, axd)):
        _kb = _key.get_window_extent(rend)
        for _pn, _pax in panels:
            if _pax is not _home:
                assert clearance_px(_kb, _pax.get_window_extent(rend)) > 0, f"the {_kname} reaches into {_pn}"
    # 3. The A key does not cover a track: its lower edge is compared with the tracks over the span it covers.
    _kd = leg_a.get_window_extent(rend).transformed(axa.transData.inverted())
    _kz0, _kz1 = min(_kd.x0, _kd.x1) / mm, max(_kd.x0, _kd.x1) / mm
    _under_key = max((float(np.max(np.abs(p[(p[:, 1] >= _kz0) & (p[:, 1] <= _kz1), 0])))
                      for p in tr["paths"] if ((p[:, 1] >= _kz0) & (p[:, 1] <= _kz1)).any()), default=0.0) * mm
    assert min(_kd.y0, _kd.y1) > _under_key, "the A key sits on the tracks"
    # 4. The D key does not cover a bar of the histogram.
    _kbd = leg_d.get_window_extent(rend)
    for _bar in axd.patches:
        assert clearance_px(_kbd, _bar.get_window_extent(rend)) > 0, "the D key covers a histogram bar"
    # 5. No two panels touch, labels included.
    _tight = [(n_, a.get_tightbbox(rend)) for n_, a in panels]
    _worst, _pair = float("inf"), None
    for _i in range(len(_tight)):
        for _j in range(_i + 1, len(_tight)):
            _d = clearance_px(_tight[_i][1], _tight[_j][1])
            if _d < _worst:
                _worst, _pair = _d, (_tight[_i][0], _tight[_j][0])
    assert _worst >= MIN_PX, f"{_pair[0]} and {_pair[1]} are {_worst:.1f} px apart, under {MIN_PX:.0f}"
    # 6. B and C are the same width.
    assert abs(axb.get_position().width - axc.get_position().width) < 1e-9
    print(f"  layout guard: tightest pair {_pair[0]}/{_pair[1]} at {_worst:.1f} px; keys clear of data")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    S.snap_panel_labels(fig)
    S.assert_keys_clear_of_data(fig)
    S.assert_ink_inside_canvas(fig)
    fig.savefig(OUT, dpi=600, facecolor="white")
    from l8_paths import flatten_png; flatten_png(OUT)
    plt.close(fig)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
