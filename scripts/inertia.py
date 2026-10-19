#!/usr/bin/env python3
"""Contribution of inertia: the 32 cases solved with and without the convective term.

Both solutions are reduced through l8_damage.derive. For every case the script prints the inertial
share of the entrance pressure, the fraction of p_en removed without the convective term
(Additional file 1, Table S1), and inertia's share of the predicted total damage. It then prints
the rise in damage from the 200 to the 330 µm nozzle at 200 kPa with and without inertia, and
compares the inertial part of p_en with the kinetic energy the jet carries out of the nozzle,
alpha * rho * u_mean**2 / 2, with alpha = 3(3n+1)**2 / ((2n+1)(5n+3)) for developed power-law flow.

    python scripts/inertia.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8_damage as D  # noqa: E402
from l8_paths import CANONICAL_GRID, DATA, require  # noqa: E402

STOKES = DATA / "fem" / "runs" / "2026-08-28_ning-grid32-stokes"


def main() -> int:
    ns = {(int(float(r["needle_um"])), float(r["pressure_kPa"])): float(r["Q_uLs"])
          for r in csv.DictReader(open(require(CANONICAL_GRID, "the 32-case grid")))}
    st = {}
    for d in sorted({k[0] for k in ns}):
        f = require(STOKES / f"ning_grid_M12_D{d}.csv", "the solves without inertia")
        for r in csv.DictReader(open(f)):
            st[(d, float(r["pressure_kPa"]))] = float(r["Q_uLs"])
    n = D.N_EXP
    alpha = 3 * (3 * n + 1) ** 2 / ((2 * n + 1) * (5 * n + 3))
    print(f"{'D':>4} {'dP':>4} {'p_en':>8} {'p_en, no inertia':>17} {'inertial share':>15} {'Dt':>7} "
          f"{'Dt, no inertia':>15} {'share of Dt':>12}")
    rows = {}
    for (d, p) in sorted(ns):
        R = d / 2 * 1e-6
        a, b = D.derive(p, ns[(d, p)], R), D.derive(p, st[(d, p)], R)
        pen_share = 100 * (a["P_en_kPa"] - b["P_en_kPa"]) / a["P_en_kPa"]
        dt_share = 100 * (a["Dt_pct"] - b["Dt_pct"]) / a["Dt_pct"]
        u = ns[(d, p)] * 1e-9 / (math.pi * R ** 2)
        ke = alpha * D.RHO * u ** 2 / 2 / 1e3
        rows[(d, p)] = (a, b, pen_share, dt_share, a["P_en_kPa"] - b["P_en_kPa"], ke)
        print(f"{d:>4} {p:>4.0f} {a['P_en_kPa']:8.3f} {b['P_en_kPa']:17.3f} {pen_share:14.1f}% "
              f"{a['Dt_pct']:7.2f} {b['Dt_pct']:15.2f} {dt_share:11.1f}%")
    print(f"\ninertial share of p_en above 50 %: {sum(v[2] > 50 for v in rows.values())} of {len(rows)} cases; "
          f"200 µm nozzle at 50-400 kPa: {min(rows[(200, p)][2] for p in (50.0, 100.0, 200.0, 400.0)):.1f} to "
          f"{max(rows[(200, p)][2] for p in (50.0, 100.0, 200.0, 400.0)):.1f} %")
    print("inertia's share of Dt at 200 kPa: " + ", ".join(
        f"{d} µm {rows[(d, 200.0)][3]:.1f} %" for d in (100, 200, 250, 330))
          + f"; largest at 330 µm: {max(rows[(330, p)][3] for p in (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0)):.1f} %")
    for k, label in ((0, "with inertia"), (1, "without inertia")):
        lo, hi = rows[(200, 200.0)][k]["Dt_pct"], rows[(330, 200.0)][k]["Dt_pct"]
        print(f"200 -> 330 µm at 200 kPa, {label}: Dt {lo:.2f} -> {hi:.2f} %, {100 * (hi - lo) / lo:+.1f} %")
    big = [v for v in rows.values() if v[4] > 0.1]
    print(f"inertial part of p_en against alpha*rho*u^2/2 (alpha = {alpha:.2f}): {len(big)} cases with an "
          f"inertial part above 0.1 kPa, agreement within {max(abs(v[4] / v[5] - 1) for v in big) * 100:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
