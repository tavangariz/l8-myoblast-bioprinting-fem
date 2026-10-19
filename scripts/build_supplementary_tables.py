#!/usr/bin/env python3
"""Per-case values for Additional file 1, Table S1.

For each of the 32 cases: the shear damage averaged over the nozzle cross-section by area (Eq. 21)
and by volumetric flux (Eq. S9), the flux-weighted fraction of the cross-section inside the
calibrated range of the shear branch (100-1000 Pa, 10-120 s), and whether the effective extensional
stress lies inside the calibrated 440-9251 Pa. Computed through l8_damage, as the grid. Writes
results/fem/exports/L8_FEM_supplementary_tables_2026-09-16.csv.

    python scripts/build_supplementary_tables.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from l8_damage import (EXT_BOX_TAU_PA, LTIP, SHEAR_BOX_TAU_PA, SHEAR_BOX_TIME_S,  # noqa: E402
                       derive, velocity_profile)
from l8_geometry import TIP_RADIUS_M                                              # noqa: E402
from l8_paths import CANONICAL_GRID, DATA, require                                # noqa: E402

OUT = DATA / "fem" / "exports" / "L8_FEM_supplementary_tables_2026-09-16.csv"


def main() -> int:
    rows = list(csv.DictReader(open(require(CANONICAL_GRID, "the canonical 32-case grid"))))
    out = []
    for r in rows:
        d = int(round(float(r["needle_um"])))
        R = TIP_RADIUS_M[d]
        P = float(r["pressure_kPa"])
        Q = float(r["Q_uLs"])
        g = derive(P, Q, R)
        # Residence time at the axis and at 97.6 % of the radius, the outermost tracked bin.
        dP_tip = g["dP_tip_kPa"] * 1e3
        t_axis = LTIP / float(velocity_profile(dP_tip, R, 0.0))
        t_edge = LTIP / float(velocity_profile(dP_tip, R, 0.976 * R))
        out.append(dict(
            needle_um=d, pressure_kPa=P,
            Dst_area_pct=g["Dst_pct"], Dst_flux_pct=g["Dst_flux_pct"],
            Dst_difference_pp=g["Dst_pct"] - g["Dst_flux_pct"],
            Dt_pct=g["Dt_pct"],
            tau_w_Pa=g["tau_w_Pa"],
            t_axis_ms=t_axis * 1e3, t_r976_ms=t_edge * 1e3,
            shear_box_flux_frac_pct=100 * g["shear_box_frac"],
            tau_ext_Pa=g["tau_ext_Pa"], ext_in_box=g["ext_in_box"]))

    cols = list(out[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out)

    print(f"wrote {len(out)} rows -> {OUT.name}\n")
    print("  BOTH SHEAR AVERAGES, percentage points\n")
    print(f"    {'tip':>5} {'kPa':>6} {'Dst_A':>9} {'Dst_Q':>9} {'A - Q':>9}")
    for r in out:
        print(f"    {r['needle_um']:>5d} {r['pressure_kPa']:>6.0f} "
              f"{r['Dst_area_pct']:9.4f} {r['Dst_flux_pct']:9.4f} "
              f"{r['Dst_difference_pp']:+9.4f}")
    worst = max(out, key=lambda r: abs(r["Dst_difference_pp"]))
    print(f"\n    largest disagreement {worst['Dst_difference_pp']:+.4f} pp at "
          f"Ø{worst['needle_um']} µm / {worst['pressure_kPa']:.0f} kPa")

    print(f"\n  NING'S CALIBRATED BOX  (shear {SHEAR_BOX_TAU_PA[0]:.0f}-{SHEAR_BOX_TAU_PA[1]:.0f} Pa"
          f" and {SHEAR_BOX_TIME_S[0]:.0f}-{SHEAR_BOX_TIME_S[1]:.0f} s;"
          f" extensional {EXT_BOX_TAU_PA[0]:.0f}-{EXT_BOX_TAU_PA[1]:.0f} Pa)\n")
    print(f"    {'tip':>5} {'kPa':>6} {'tau_w':>9} {'t_axis':>9} {'t@.976R':>9}"
          f" {'in box':>8} {'tau_e':>10} {'in box':>7}")
    for r in out:
        print(f"    {r['needle_um']:>5d} {r['pressure_kPa']:>6.0f} {r['tau_w_Pa']:9.1f}"
              f" {r['t_axis_ms']:8.2f}m {r['t_r976_ms']:8.2f}m"
              f" {r['shear_box_flux_frac_pct']:7.1f}%"
              f" {r['tau_ext_Pa']:10.1f} {'yes' if r['ext_in_box'] else 'NO':>7}")
    n_ext_out = sum(1 for r in out if not r["ext_in_box"])
    n_any = sum(1 for r in out if r["shear_box_flux_frac_pct"] > 0)
    lo = min(r["t_axis_ms"] for r in out) / 1e3
    hi = max(r["t_r976_ms"] for r in out) / 1e3
    print(f"\n    shear box: {n_any} of {len(out)} cases have ANY part of the cross-section "
          f"jointly inside it; {len(out) - n_any} are wholly outside")
    print(f"    residence time spans {lo*1e3:.2f} ms to {hi:.0f} s across the grid, against a "
          f"fitted window of {SHEAR_BOX_TIME_S[0]:.0f}-{SHEAR_BOX_TIME_S[1]:.0f} s — the fast "
          f"cases fall BELOW the floor and the slowest Ø100 cases rise ABOVE the ceiling")
    print(f"    at the reference condition (Ø200, 200 kPa) the fraction is "
          f"{[r for r in out if r['needle_um']==200 and r['pressure_kPa']==200][0]['shear_box_flux_frac_pct']:.1f} %")
    print(f"    extensional box: {n_ext_out} of {len(out)} cases fall outside "
          f"{EXT_BOX_TAU_PA[0]:.0f}-{EXT_BOX_TAU_PA[1]:.0f} Pa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
