#!/usr/bin/env python3
"""The tracer paths of Fig. 5 and Table S11 against the integration tolerances (Additional file 1, Section S10).

A tracer released near the cartridge wall passes within a few µm of the contraction corner, and its radius in the
nozzle is the slowest quantity of the tracking to converge. For each of the four sets of tracking_sensitivity.py,
the kept tracers of largest radius in the nozzle are re-tracked from their release radii with the same field,
integrator and tests, at the tolerances of the tracking and at 100-fold tighter ones. Writes
results/analyses/tracking_convergence.json: per set and over all sets, the largest change of the radius in the
nozzle and of the shear damage of Eq. 20 between the two, and the largest difference from tracking_tracers.csv.

    python scripts/tracking_convergence.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.interpolate import LinearNDInterpolator

import fig5_lagrangian_tracers as F
import tracking_sensitivity as T
from l8_paths import DATA

N_OUTER = 20      # tracers per set: those of largest radius in the nozzle
TIGHTER = 100.0   # factor on both tolerances


def release_radii(raw):
    """Release radii of the four sets, drawn exactly as tracking_sensitivity.main draws them."""
    g = raw["geom"]
    rng = np.random.default_rng(F.RNG_SEED)
    out = {"published": 0.95 * g["Rin"] * np.sqrt(rng.uniform(0, 1, int(F.N_TRACERS * 4.0)))}
    rng = np.random.default_rng(1)
    for sid, frac in (("pool_0.95", 0.95), ("pool_0.99", 0.99)):
        out[sid] = frac * g["Rin"] * np.sqrt(rng.uniform(0, 1, T.POOL))
    iz = LinearNDInterpolator(raw["pts2d"], raw["u"][:, 1], fill_value=np.nan)
    rr = np.linspace(0, 0.99 * g["Rin"], 400)
    uz = np.nan_to_num(iz(np.column_stack([rr, np.full_like(rr, F.Z_SEED)])), nan=0.0)
    seeds = []
    while len(seeds) < T.POOL:
        r = 0.99 * g["Rin"] * np.sqrt(rng.uniform())
        if rng.uniform() < np.interp(r, rr, uz) / uz.max():
            seeds.append(r)
    out["flux_0.99"] = np.array(seeds)
    return out


def main():
    raw = F.load_checkpoint(F.CKPT)
    ds_at = F.shear_damage_function(raw["geom"])
    rows = list(csv.DictReader(open(DATA / "analyses" / "tracking_tracers.csv")))
    base = (F.RTOL, F.ATOL)
    tight = (F.RTOL / TIGHTER, F.ATOL / TIGHTER)
    out = dict(rtol=base[0], atol=base[1], rtol_tighter=tight[0], atol_tighter=tight[1], n_per_set=N_OUTER, sets={})
    for sid, pool in release_radii(raw).items():
        outer = sorted((r for r in rows if r["set"] == sid), key=lambda r: -float(r["nozzle_r_um"]))[:N_OUTER]
        r0 = []
        for k in outer:
            j = int(np.argmin(np.abs(1e3 * pool - float(k["release_r_mm"]))))
            if abs(1e3 * pool[j] - float(k["release_r_mm"])) > 1e-6:
                raise SystemExit(f"{sid}: no drawn seed at {k['release_r_mm']} mm; the draws differ from the tracking")
            r0.append(float(pool[j]))
        res = {}
        for tag, (rt, at) in (("tracking", base), ("tighter", tight)):
            F.RTOL, F.ATOL = rt, at
            kept, _ = T.track(raw, ds_at, np.array(r0))
            res[tag] = {k["r0"]: k for k in kept}
        F.RTOL, F.ATOL = base
        both = [r for r in r0 if r in res["tracking"] and r in res["tighter"]]
        a, b = res["tracking"], res["tighter"]
        csv_by_r0 = {round(1e3 * r, 6): k for r, k in zip(r0, outer)}
        out["sets"][sid] = dict(
            n=len(r0), n_kept_both=len(both),
            max_dr_um=max(1e6 * abs(a[r]["r_noz"] - b[r]["r_noz"]) for r in both),
            max_dDs_pp=max(100 * abs(a[r]["D_s"] - b[r]["D_s"]) for r in both),
            max_dr_vs_table_um=max(abs(1e6 * a[r]["r_noz"] - float(csv_by_r0[round(1e3 * r, 6)]["nozzle_r_um"]))
                                   for r in both),
            max_dDs_vs_table_pp=max(abs(100 * a[r]["D_s"] - float(csv_by_r0[round(1e3 * r, 6)]["D_s_pct"]))
                                    for r in both))
        print(sid, out["sets"][sid], flush=True)
    s = out["sets"].values()
    out.update(n=sum(x["n"] for x in s), n_kept_both=sum(x["n_kept_both"] for x in s),
               max_dr_um=max(x["max_dr_um"] for x in s), max_dDs_pp=max(x["max_dDs_pp"] for x in s))
    (DATA / "analyses" / "tracking_convergence.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "sets"})


if __name__ == "__main__":
    main()
