#!/usr/bin/env python3
"""Data readers for Fig. 3C and Additional file 1, Fig. S2.

tracer_entry()   shear rate and radius of each tracer at nozzle entry, from the solved field
read_pairs()     the 32 cases on meshes M12 and M16, reduced through l8_damage.derive
residual_rows()  entrance pressure against a Bagley extrapolation, for each nozzle diameter
"""
from __future__ import annotations

import csv
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

from l8_damage import derive                       # noqa: E402
from l8_geometry import GEOM, TIP_RADIUS_M, wall_radius   # noqa: E402
from l8_paths import CANONICAL_GRID, CHECKPOINT, DATA, require  # noqa: E402

TIPS = (100, 200, 250, 330)

# Tracer_entry() — Lagrangian tracers, shear rate and radius in the nozzle
_K_VIS, _N_EXP = 25.60, 0.35
_N_CELLS_DEFAULT = 440
_RNG_SEED = 42
_Z_SEED_DEFAULT = 2e-4
_READ_DEPTH_DIAMETERS = 1.0   # the reading plane, in nozzle diameters past the inlet plane z2


def _load_ckpt_fields(path):
    import pickle
    ck = pickle.load(open(path, "rb"))
    geom = {k: ck[k] for k in ("Rin", "Rmid", "Rtip", "z1", "z2", "z3")}
    return dict(pts2d=np.asarray(ck["vtk_coords"]), u=np.asarray(ck["fields"]["uvec"]),
                gdot=np.asarray(ck["fields"]["gdot"]), tau_w=float(ck["tau_w"]), geom=geom)


def _integrate_to_outlet(raw, n_cells, z_seed):
    """Track `n_cells` tracers, seeded uniform by area at z = z_seed, to the outlet.

    The ODE, the tolerances (`fig5_lagrangian_tracers.RTOL`, `ATOL`), the outlet event and the
    wall test are those of the Fig. 5 tracking; only the release plane (0.2 mm into the cartridge)
    and the seed count differ. Returns the paths that reach the outlet inside the bore.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import LinearNDInterpolator
    import fig5_lagrangian_tracers as F

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
        return [0.0 if not np.isfinite(ur) else ur, 0.0 if not np.isfinite(uz) else uz]

    rng = np.random.default_rng(_RNG_SEED)
    r_seeds = 0.95 * g["Rin"] * np.sqrt(rng.uniform(0, 1, n_cells))

    paths = []
    for i in range(n_cells):
        sol = solve_ivp(ode, (0.0, 120.0), [float(r_seeds[i]), z_seed], events=outlet,
                        max_step=0.01, rtol=F.RTOL, atol=F.ATOL)
        path = sol.y[:2].T
        wall_ok = np.all(np.abs(path[:, 0]) <= wall_radius(path[:, 1], g) + 3e-6)
        if path.shape[0] <= 5 or not wall_ok or path[-1, 1] < g["z3"] * 0.999:
            continue
        paths.append(path)
    return paths


def _radius_at(path, z, rmax):
    """|r| where a tracer crosses the plane z: a cubic in z through the two stored path points on either side of it,
    as `fig5_lagrangian_tracers.entry_radius` reads the inlet plane."""
    j = int(np.argmax(path[:, 1] >= z))
    k = slice(max(j - 2, 0), min(j + 2, len(path)))
    zz, rr = path[k, 1] - z, path[k, 0]
    return float(min(abs(np.polyval(np.polyfit(zz, rr, min(3, len(zz) - 1)), 0.0)), rmax))


def tracer_entry(stats=None):
    """Radius and shear rate in the nozzle for every tracer of the entry set, read one nozzle diameter past the inlet.

    On the inlet plane z = z2 itself the strain rate of the solution is dominated by the extension of the
    contraction, and it is singular at the re-entrant corner, so a reading there depends on where it is taken. One
    nozzle diameter downstream the velocity profile has developed. Both are checked on every run and written into
    `stats`: the ratio of each reading to the developed power-law profile gdot_wall (r/R)^(1/n) at the same radius,
    and the change of the median when the plane moves to 2.5 diameters. The radius is read where each path crosses
    the plane, the radius at which the tracer travels through the nozzle. A tracer whose shear rate exceeds the analytic wall value would be a reading of the corner singularity and
    is dropped; at this plane none does. If `stats` is a dict, the seeded, arrived, dropped and kept counts, the
    reading depth and the wall value are written into it.
    """
    import fig5_lagrangian_tracers as F
    require(CHECKPOINT, "the solver checkpoint")
    raw = _load_ckpt_fields(str(CHECKPOINT))
    g = raw["geom"]
    from scipy.interpolate import LinearNDInterpolator
    gd_interp = LinearNDInterpolator(raw["pts2d"], raw["gdot"], fill_value=np.nan)

    paths = _integrate_to_outlet(raw, _N_CELLS_DEFAULT, _Z_SEED_DEFAULT)

    z_read = g["z2"] + _READ_DEPTH_DIAMETERS * 2.0 * g["Rtip"]
    assert z_read < g["z3"], "the reading plane lies beyond the nozzle outlet"
    r_e = np.array([_radius_at(p_, z_read, g["Rtip"]) for p_ in paths])
    g_e = np.array([float(gd_interp([[r, z_read]])[0]) for r in r_e])

    # The cap is derived from the checkpoint's own tau_w, not typed.
    gdot_wall = (raw["tau_w"] / _K_VIS) ** (1.0 / _N_EXP)
    keep = np.isfinite(g_e) & (g_e <= gdot_wall)
    n_drop = int((~keep).sum())
    print(f"    {len(paths)} of {_N_CELLS_DEFAULT} tracers reach the outlet; read {1e6 * (z_read - g['z2']):.0f} um past "
          f"the inlet; {n_drop} above gdot_wall = {gdot_wall:.4g} 1/s; {int(keep.sum())} kept")
    # Developed-flow checks: the reading against the power-law profile, and against a plane 2.5 diameters in
    ratio = g_e[keep] / (gdot_wall * (r_e[keep] / g["Rtip"]) ** (1.0 / _N_EXP))
    z_far = g["z2"] + 2.5 * 2.0 * g["Rtip"]
    r_far = np.array([_radius_at(p_, z_far, g["Rtip"]) for p_ in paths])
    g_far = np.array([float(gd_interp([[r, z_far]])[0]) for r in r_far])
    shift = 100.0 * (np.median(g_far[keep]) / np.median(g_e[keep]) - 1.0)
    q05, q50, q95 = np.quantile(ratio, [0.05, 0.5, 0.95])
    print(f"    reading / developed profile: median {q50:.3f}, 5-95% {q05:.3f}-{q95:.3f}; median at 2.5 diameters "
          f"{shift:+.2f}%")
    if stats is not None:
        stats.update(seeded=_N_CELLS_DEFAULT, reached_outlet=len(paths), dropped_above_wall=n_drop,
                     kept=int(keep.sum()), gdot_wall_per_s=gdot_wall, release_z_mm=1e3 * _Z_SEED_DEFAULT,
                     read_depth_um=1e6 * (z_read - g["z2"]), ode_rtol=F.RTOL, ode_atol=F.ATOL,
                     developed_ratio_median=float(q50), developed_ratio_p05=float(q05), developed_ratio_p95=float(q95),
                     median_shift_at_2p5_diameters_pct=float(shift))
    return r_e[keep], g_e[keep]


# Read_pairs() — M12/M16 mesh-convergence pairing, 32 cases
_R_TIP = TIP_RADIUS_M
_GRID_DIR = DATA / "fem" / "runs" / "2026-08-26_ning-grid32"


def _col(path):
    require(path, f"solver output {path.name}")
    return {float(r["pressure_kPa"]): r for r in csv.DictReader(open(path))}


def read_pairs():
    """32 cases, each solved at M12 and M16. Returns per-tip pressure / |dD_t| / |dQ|
    arrays, plus the headline's own M12 and M16 rise, all reduced through `l8_damage.derive`
    — the same chain `build_canonical_grid.py` uses to write the canonical CSV."""
    head = {}
    for M in (12, 16):
        lo = _col(_GRID_DIR / f"ning_grid_M{M}_D200.csv")
        hi = _col(_GRID_DIR / f"ning_grid_M{M}_D330.csv")
        d2 = derive(200.0, float(lo[200.0]["Q_uLs"]), _R_TIP[200])["Dn_pct"]
        d3 = derive(200.0, float(hi[200.0]["Q_uLs"]), _R_TIP[330])["Dn_pct"]
        head[M] = 100 * (d3 - d2) / d2

    out, dt_all, dq_all = {}, [], []
    for d in TIPS:
        a = _col(_GRID_DIR / f"ning_grid_M12_D{d}.csv")
        b = _col(_GRID_DIR / f"ning_grid_M16_D{d}.csv")
        shared = sorted(set(a) & set(b))
        p, ddt = [], []
        for pk in shared:
            qa, qb = float(a[pk]["Q_uLs"]), float(b[pk]["Q_uLs"])
            da = derive(pk, qa, _R_TIP[d])["Dn_pct"]
            db = derive(pk, qb, _R_TIP[d])["Dn_pct"]
            p.append(pk)
            ddt.append(abs(db - da))
            dt_all += [da, db]
            dq_all.append(abs(100 * (qb - qa) / qa))
        out[d] = (np.array(p), np.array(ddt))
    return out, np.array(dt_all), np.array(dq_all), head


# Residual_rows() — reported P_en against an independent Bagley extrapolation
_BAGLEY_DIR = DATA / "fem" / "runs" / "2026-08-27_bagley-ning"
_BAGLEY_L = {"2p5mm": 2.5, "5mm": 5.0, "10mm": 10.0}
_R_BARREL, _L_BARREL = GEOM["Rin"], GEOM["z1"]              # straight cartridge
_R_MID, _L_MID = GEOM["Rmid"], GEOM["z2"] - GEOM["z1"]       # funnel exit radius, length


def _target_q():
    """The production flow rate each tip's Bagley series is interpolated to: the 200 kPa,
    M12 row of the canonical grid, read rather than hardcoded so it cannot go stale under a
    domain correction."""
    require(CANONICAL_GRID, "the canonical 32-case grid")
    out = {}
    for r in csv.DictReader(open(CANONICAL_GRID)):
        if float(r["pressure_kPa"]) == 200.0 and float(r["mesh_M"]) == 12.0:
            out[int(float(r["needle_um"]))] = round(float(r["Q_uLs"]), 4)
    if not out:
        raise SystemExit(f"FAIL: no 200 kPa / M12 rows in {CANONICAL_GRID}")
    return out


def _read_bagley_series(name):
    return _read_bagley_series_at(_BAGLEY_DIR / name)


def _read_bagley_series_at(path):
    require(path, f"Bagley series {path.name}")
    with open(path) as fh:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def _dP_developed(Q_m3s, R, L):
    """Fully-developed power-law pressure drop in a straight tube (Pa). Rabinowitsch."""
    gdot_w = (3 * _N_EXP + 1) / (4 * _N_EXP) * 4 * Q_m3s / (math.pi * R ** 3)
    return 2 * L * _K_VIS * gdot_w ** _N_EXP / R


def _dP_cone(Q_m3s, R_in, R_out, L):
    """Power-law pressure drop along a linearly tapering cone (Pa)."""
    C = (3 * _N_EXP + 1) / (4 * _N_EXP)
    pref = 2 * _K_VIS * (C * 4 * Q_m3s / math.pi) ** _N_EXP
    n3 = 3 * _N_EXP
    return pref * L / (n3 * (R_in - R_out)) * (R_out ** -n3 - R_in ** -n3)


def _feed_friction_kPa(Q_uLs):
    """Fully-developed friction in the straight cartridge plus the chamfer cone (kPa)."""
    Q = Q_uLs * 1e-9
    return (_dP_developed(Q, _R_BARREL, _L_BARREL)
            + _dP_cone(Q, _R_BARREL, _R_MID, _L_MID)) / 1e3


def _linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    return slope, icpt


def _loglog_interp(rows, q_target):
    """Interpolate applied pressure to a target flow rate, log-log (exact for a power-law
    fluid, whose flow rate is a power of the pressure drop)."""
    pts = sorted((math.log(r["Q_uLs"]), math.log(r["pressure_kPa"])) for r in rows)
    slope, icpt = _linfit([p[0] for p in pts], [p[1] for p in pts])
    return math.exp(icpt + slope * math.log(q_target))


def residual_rows():
    """(needle, bagley_kPa, naive_residual_kPa, friction_kPa, reported_kPa) per tip.

    `reported_kPa` is the canonical grid's own entrance pressure (the analytic-tau_w
    residual every P_en/tau_e/D_et in the manuscript comes from); `naive_residual_kPa` is the
    raw-extracted-tau_w residual route that no reported value uses, kept here only as
    the comparison this panel exists to draw.
    """
    target_q = _target_q()
    data = {tag: [r for d in TIPS
                  for r in _read_bagley_series(f"bagley_ning_D{d}_L{tag}.csv")]
            for tag in _BAGLEY_L}

    rows = []
    for needle in TIPS:
        qt = target_q[needle]
        xs, ys = [], []
        for tag, L in sorted(_BAGLEY_L.items(), key=lambda t: t[1]):
            series = [r for r in data[tag] if r["needle_um"] == needle]
            xs.append(L)
            ys.append(_loglog_interp(series, qt))
        _, icpt = _linfit(xs, ys)

        prod = [r for r in data["10mm"] if r["needle_um"] == needle]
        prod_row = min(prod, key=lambda r: abs(r["Q_uLs"] - qt))
        # The analytic-tau_w residual route, evaluated on the Bagley cluster's own 200 kPa production
        # solve — a separate job from the canonical grid's, so it disagrees with `rep` below by more
        # than solve noise where the two runs' converged flow rates differ.
        P_res = prod_row["P_en_kPa"]
        fric = _feed_friction_kPa(prod_row["Q_uLs"])
        # The reported value: the same analytic route, at the canonical grid's own converged flow rate
        # for this tip (target_q, read in _target_q()), which is what every P_en / tau_e / D_et in the
        # manuscript comes from.
        rep = derive(200.0, qt, TIP_RADIUS_M[needle])["P_en_kPa"]
        rows.append((needle, icpt, P_res, fric, rep))
    return sorted(rows)
