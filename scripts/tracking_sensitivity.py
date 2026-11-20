#!/usr/bin/env python3
"""How the tracer statistics of Fig. 5 depend on the tracer set (Additional file 1, Section S10, Table S11).

Fig. 5 keeps the first 168 tracers, of a seeded pool, that reach the outlet inside the bore. A tracer is
dropped if its path does not reach the outlet within the integration window or if it crosses the wall by more
than 3 µm (fig5_lagrangian_tracers.integrate). Each kept tracer takes the shear damage of Eq. 20 at the radius
it travels at through the straight nozzle. The tracking is repeated at the reference condition with the same
field, ODE and tests, and for each tracer set the script reports:

  how many tracers were seeded, kept and dropped, and why (a tracer that ends on the contraction face,
  outside the bore, where the interpolated velocity is zero, is reported as such), with the release
  radii of the dropped ones;
  the outermost radius reached in the nozzle;
  the mean, 95th percentile and spread of the per-tracer shear damage, beside the area average of
  Eq. 21 and the flux-weighted average (Additional file 1, Eq. S9).

Sets: the one of Fig. 5 (first 168 survivors, 0.95 R_in, area-uniform); every survivor of a pool of 1000 at
0.95 R_in; 0.99 R_in; flux-weighted release (seed density proportional to u_z on the release plane), which
samples the cells in proportion to how many pass.

Writes results/analyses/tracking_sensitivity.json, every kept tracer (set, release radius, nozzle-entry
radius, radius in the nozzle, shear damage) to results/analyses/tracking_tracers.csv, which Fig. 5 plots, and
the paths of the Fig. 5 set to a cache keyed by the checkpoint's md5, so the figure need not re-track them.

    python scripts/tracking_sensitivity.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator

import fig5_lagrangian_tracers as F
from l8_paths import DATA

POOL = 1000


def track(raw, ds_at, r_seeds, keep_first=None, paths=None):
    g = raw["geom"]
    ir = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 0], fill_value=np.nan)
    iz = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 1], fill_value=np.nan)

    def outlet(_t, y):
        return y[1] - g["z3"]
    outlet.terminal, outlet.direction = True, 1

    def ode(_t, y):
        q = [[abs(y[0]), y[1]]]
        ur, uz = float(ir(q)[0]), float(iz(q)[0])
        return [0.0 if not np.isfinite(ur) else ur * (1.0 if y[0] >= 0 else -1.0), 0.0 if not np.isfinite(uz) else uz]

    kept, dropped = [], []
    for r0 in r_seeds:
        if keep_first is not None and len(kept) >= keep_first:
            break
        sol = solve_ivp(ode, (0.0, 120.0), [float(r0), F.Z_SEED], events=outlet, max_step=0.01, rtol=F.RTOL, atol=F.ATOL)
        path = sol.y[:2].T
        if path.shape[0] <= 5 or path[-1, 1] < g["z3"] * 0.999:
            on_face = abs(path[-1, 1] - g["z2"]) < 5e-6 and abs(path[-1, 0]) > g["Rtip"]
            dropped.append(dict(r0_mm=1e3 * float(r0), why="stopped on the contraction face" if on_face
                                else "did not reach the outlet within 120 s",
                                end_z_mm=round(1e3 * float(path[-1, 1]), 4), end_r_um=round(1e6 * abs(float(path[-1, 0])), 2)))
            continue
        if not np.all(np.abs(path[:, 0]) <= F.radius_at_z(path[:, 1], g) + 3e-6):
            dropped.append(dict(r0_mm=1e3 * float(r0), why="crossed the wall"))
            continue
        in_tip = path[:, 1] >= g["z2"]
        if not np.any(in_tip):
            dropped.append(dict(r0_mm=1e3 * float(r0), why="never entered the nozzle"))
            continue
        r_tip = F.entry_radius(path, g)
        r_noz = F.nozzle_radius(path, g)
        kept.append(dict(r0=float(r0), r_tip=r_tip, r_noz=r_noz, D_s=float(ds_at(r_noz))))
        if paths is not None:
            paths.append(path)
    return kept, dropped


def summarise(name, kept, dropped, prof, g):
    ds = 100 * np.array([k["D_s"] for k in kept])
    rt = np.array([k["r_noz"] for k in kept]) / g["Rtip"]
    dt = 100 * prof["D_e"] + (1 - prof["D_e"]) * ds
    return dict(set=name, seeded=len(kept) + len(dropped), kept=len(kept), dropped=len(dropped),
                dropped_by_reason={w: sum(d["why"] == w for d in dropped) for w in {d["why"] for d in dropped}},
                dropped_release_radius_mm=sorted(round(d["r0_mm"], 3) for d in dropped),
                dropped_end_points=[(d["end_z_mm"], d["end_r_um"]) for d in dropped],
                outermost_nozzle_r_over_R=round(float(rt.max()), 4),
                mean_Dst_pct=round(float(ds.mean()), 4), p95_Dst_pct=round(float(np.percentile(ds, 95)), 4),
                max_Dst_pct=round(float(ds.max()), 4), Dt_spread_pp=round(float(dt.max() - dt.min()), 4))


def main():
    raw = F.load_checkpoint(F.CKPT)
    prof = F.compute_damage_profile(raw)
    g = raw["geom"]
    ds_at = F.shear_damage_function(g)
    rng = np.random.default_rng(F.RNG_SEED)
    pub = 0.95 * g["Rin"] * np.sqrt(rng.uniform(0, 1, int(F.N_TRACERS * 4.0)))
    res, rows, pub_paths = [], [], []

    def record(set_id, name, kept, dropped):
        res.append(summarise(name, kept, dropped, prof, g))
        rows.extend((set_id, round(1e3 * k["r0"], 6), round(1e6 * k["r_tip"], 6), round(1e6 * k["r_noz"], 6),
                     round(100 * k["D_s"], 8)) for k in kept)

    kept, dropped = track(raw, ds_at, pub, keep_first=F.N_TRACERS, paths=pub_paths)
    record("published", "published: first 168 survivors, 0.95 R_in", kept, dropped)
    rng = np.random.default_rng(1)
    for set_id, name, frac in (("pool_0.95", "all survivors of 1000, 0.95 R_in", 0.95),
                               ("pool_0.99", "all survivors of 1000, 0.99 R_in", 0.99)):
        seeds = frac * g["Rin"] * np.sqrt(rng.uniform(0, 1, POOL))
        k, d = track(raw, ds_at, seeds)
        record(set_id, name, k, d)
    # flux-weighted release: accept r with probability u_z(r, Z_SEED) / max u_z on the plane
    iz = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 1], fill_value=np.nan)
    rr = np.linspace(0, 0.99 * g["Rin"], 400)
    uz = np.nan_to_num(iz(np.column_stack([rr, np.full_like(rr, F.Z_SEED)])), nan=0.0)
    seeds = []
    while len(seeds) < POOL:
        r = 0.99 * g["Rin"] * np.sqrt(rng.uniform())
        if rng.uniform() < np.interp(r, rr, uz) / uz.max():
            seeds.append(r)
    k, d = track(raw, ds_at, np.array(seeds))
    record("flux_0.99", "all survivors of 1000, flux-weighted, 0.99 R_in", k, d)
    out = dict(area_average_Dst_pct=round(100 * prof["D_st"], 4), D_e_pct=round(100 * prof["D_e"], 4),
               readout="shear damage of Eq. 20 at the median radius over the nozzle beyond its first 1 mm",
               ode_rtol=F.RTOL, ode_atol=F.ATOL, sets=res)
    (DATA / "analyses").mkdir(parents=True, exist_ok=True)
    (DATA / "analyses" / "tracking_sensitivity.json").write_text(json.dumps(out, indent=1))
    with open(DATA / "analyses" / "tracking_tracers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "release_r_mm", "entry_r_um", "nozzle_r_um", "D_s_pct"])
        w.writerows(rows)
    md5 = hashlib.md5(open(F.CKPT, "rb").read()).hexdigest()
    np.savez_compressed(DATA / "analyses" / "tracking_published_paths_cache.npz", ckpt_md5=md5, rtol=F.RTOL, atol=F.ATOL,
                        xy=np.concatenate(pub_paths), ptr=np.cumsum([0] + [len(p) for p in pub_paths]))
    print(json.dumps(out, indent=1)[:4000])
    print(f"wrote {len(rows)} tracers and {len(pub_paths)} published paths (checkpoint md5 {md5})")


if __name__ == "__main__":
    main()
