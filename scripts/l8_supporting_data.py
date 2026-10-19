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

# Tracer_entry() — Lagrangian tracers, shear rate and radius at needle entry
_K_VIS, _N_EXP = 25.60, 0.35
_N_WANT = 168
# 440 seeded, 168 wanted: seeding beyond the wanted count and truncating keeps the ensemble off the
# re-entrant corner singularity while still reaching 168 physical survivors — see the gdot_wall cut
# below.
_N_CELLS_DEFAULT = 440
_RNG_SEED = 42
_Z_SEED_DEFAULT = 2e-4


def _load_ckpt_fields(path):
    import pickle
    ck = pickle.load(open(path, "rb"))
    geom = {k: ck[k] for k in ("Rin", "Rmid", "Rtip", "z1", "z2", "z3")}
    return dict(pts2d=np.asarray(ck["vtk_coords"]), u=np.asarray(ck["fields"]["uvec"]),
                gdot=np.asarray(ck["fields"]["gdot"]), tau_w=float(ck["tau_w"]), geom=geom)


def _integrate_to_outlet(raw, n_cells, z_seed):
    """Track `n_cells` tracers, seeded uniform by area at z = z_seed, to the outlet.

    Identical integration to `fig5_lagrangian_tracers.py`'s own tracking (same ODE, same
    outlet event, same wall test), except every seed is kept rather than only the first
    n_want survivors, and the seed count and release plane are the ones the source script
    used for this panel (440 tracers released at z = 0.2 mm), not the ones Fig. 3 uses (168
    wanted from a 4x pool, released at z = 5 mm).
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import LinearNDInterpolator

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
    th_seeds = rng.uniform(0.0, 2.0 * np.pi, n_cells)   # drawn to keep the RNG stream aligned

    paths = []
    for i in range(n_cells):
        sol = solve_ivp(ode, (0.0, 120.0), [float(r_seeds[i]), z_seed], events=outlet,
                         max_step=0.01, rtol=1e-4, atol=1e-9)
        path = sol.y[:2].T
        wall_ok = np.all(np.abs(path[:, 0]) <= wall_radius(path[:, 1], g) + 3e-6)
        if path.shape[0] <= 5 or not wall_ok or path[-1, 1] < g["z3"] * 0.999:
            continue
        paths.append(path)
    _ = th_seeds  # consumed only to keep the RNG draw sequence identical to the source
    return paths


def tracer_entry():
    """gdot and radius at tip entry for the 168 physical tracers.

    Drops any tracer whose recorded entry
    shear exceeds the analytic wall value — a re-entrant-corner-singularity artefact, not a
    resolved sample — before truncating to the 168 the manuscript reports.
    """
    require(CHECKPOINT, "the solver checkpoint")
    raw = _load_ckpt_fields(str(CHECKPOINT))
    g = raw["geom"]
    from scipy.interpolate import LinearNDInterpolator
    gd_interp = LinearNDInterpolator(raw["pts2d"], raw["gdot"], fill_value=np.nan)

    paths = _integrate_to_outlet(raw, _N_CELLS_DEFAULT, _Z_SEED_DEFAULT)

    r_e, g_e = [], []
    for p_ in paths:
        r, z = np.abs(p_[:, 0]), p_[:, 1]
        i = int(np.argmax(z >= g["z2"]))
        if z[i] >= g["z2"]:
            r_e.append(min(r[i], g["Rtip"]))
            g_e.append(float(gd_interp([[r[i], z[i]]])[0]))
    r_e, g_e = np.asarray(r_e), np.asarray(g_e)

    # A tracer's shear rate as it enters the tip cannot physically exceed the wall value there; a
    # sample above it is reading the re-entrant corner singularity.
    gdot_wall = (raw["tau_w"] / _K_VIS) ** (1.0 / _N_EXP)
    keep = g_e <= gdot_wall
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"    dropped {n_drop} corner-singularity tracer(s) above "
              f"gdot_wall = {gdot_wall:.4g} 1/s")
    r_e, g_e = r_e[keep], g_e[keep]
    if len(r_e) < _N_WANT:
        raise SystemExit(f"only {len(r_e)} physical tracers reached the tip; widen "
                         f"the seed pool beyond {_N_CELLS_DEFAULT} to reach {_N_WANT}")
    return r_e[:_N_WANT], g_e[:_N_WANT]


# Read_pairs() — M12/M16 mesh-convergence pairing, 32 cases
_R_TIP = {100: 50e-6, 200: 100e-6, 250: 125e-6, 330: 165e-6}
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
        d2 = derive(200.0, float(lo[200.0]["Q_uLs"]), _R_TIP[200])["Dt_pct"]
        d3 = derive(200.0, float(hi[200.0]["Q_uLs"]), _R_TIP[330])["Dt_pct"]
        head[M] = 100 * (d3 - d2) / d2

    out, dt_all, dq_all = {}, [], []
    for d in TIPS:
        a = _col(_GRID_DIR / f"ning_grid_M12_D{d}.csv")
        b = _col(_GRID_DIR / f"ning_grid_M16_D{d}.csv")
        shared = sorted(set(a) & set(b))
        p, ddt = [], []
        for pk in shared:
            qa, qb = float(a[pk]["Q_uLs"]), float(b[pk]["Q_uLs"])
            da = derive(pk, qa, _R_TIP[d])["Dt_pct"]
            db = derive(pk, qb, _R_TIP[d])["Dt_pct"]
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
    path = _BAGLEY_DIR / name
    require(path, f"Bagley series {name}")
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
