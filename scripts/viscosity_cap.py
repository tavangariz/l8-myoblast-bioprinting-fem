#!/usr/bin/env python3
"""How much the upper viscosity bound changes the computed flow, over the 32 cases (Additional file 1, Section S1).

The solver bounds the power-law viscosity between 0.003 and 500 Pa s. The upper bound acts only where the shear
rate is below gc = (500/K)**(1/(n-1)), that is where the shear stress is below 500 gc, near the axis of the
cartridge, where the flow is slow. The volume share where it acts depends on how well a coarse cartridge mesh
resolves the shear rate near the axis, so the effect is bounded instead: at each case's solved flow rate, the
developed pressure drop of the cartridge (radius Rin, length z1) and of the cone (taken as a tube at its outlet
radius Rmid, length z2 - z1) is computed with and without the bound. In developed flow the stress is set by the
pressure gradient, tau = G r / 2, so only the shear rate inside the bounded core changes. The difference, as a
share of the applied pressure, is the extra pressure the nozzle receives, and the flow rate of a power-law nozzle
scales as pressure**(1/n), so the flow changes by about that share divided by n.

Writes results/analyses/viscosity_cap.json.

    python scripts/viscosity_cap.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from l8_damage import K_VIS, N_EXP
from l8_geometry import GEOM
from l8_paths import CANONICAL_GRID, REPO

ETA_MAX = 500.0                                   # the solver's upper viscosity bound (Methods)
GC = (ETA_MAX / K_VIS) ** (1 / (N_EXP - 1))       # shear rate below which the bound acts
TC = ETA_MAX * GC                                 # the shear stress at that shear rate
UPSTREAM = [(GEOM["Rin"], GEOM["z1"]), (GEOM["Rmid"], GEOM["z2"] - GEOM["z1"])]   # cartridge, cone at its outlet


def flow(G, R, cap):
    """Developed flow rate of a tube of radius R under pressure gradient G, with or without the bound."""
    def gd(t):
        return t / ETA_MAX if (cap and t < TC) else (t / K_VIS) ** (1 / N_EXP)
    knot = 2 * TC / G
    return np.pi * quad(lambda s: gd(G * s / 2) * s ** 2, 0, R, points=[knot] if knot < R else None, limit=200)[0]


def drop(Q, cap):
    return sum(brentq(lambda G: flow(G, R, cap) - Q, 1e-9, 1e12, xtol=1e-14, rtol=1e-12) * L for R, L in UPSTREAM)


def main():
    rows = []
    for r in csv.DictReader(open(CANONICAL_GRID)):
        Q, P = float(r["Q_uLs"]) * 1e-9, float(r["pressure_kPa"]) * 1e3
        share = (drop(Q, False) - drop(Q, True)) / P
        rows.append(dict(D_um=float(r["needle_um"]), P_kPa=float(r["pressure_kPa"]),
                         pressure_pct=100 * share, flow_pct=100 * share / N_EXP))
    worst = max(rows, key=lambda x: x["pressure_pct"])
    ref = next(x for x in rows if (x["D_um"], x["P_kPa"]) == (200.0, 200.0))
    out = dict(eta_max_Pa_s=ETA_MAX, shear_rate_below_s=GC, shear_stress_below_Pa=TC, n_cases=len(rows),
               largest=worst, reference=ref, cases=rows)
    path = REPO / "results" / "analyses" / "viscosity_cap.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"bound acts where gdot < {GC:.4f} 1/s, shear stress < {TC:.2f} Pa")
    print(f"{len(rows)} cases; largest: {worst['D_um']:.0f} um, {worst['P_kPa']:.0f} kPa: "
          f"{worst['pressure_pct']:.4f}% of the applied pressure, flow rate ~{worst['flow_pct']:.3f}%")
    print(f"reference 200 um, 200 kPa: {ref['pressure_pct']:.2e}% of the pressure")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
