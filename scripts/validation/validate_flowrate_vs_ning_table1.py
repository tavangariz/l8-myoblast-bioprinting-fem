#!/usr/bin/env python3
"""Solved flow rate against the mass flow rates weighed by Ning et al. (2018), Table 1.

The L8 suspension through the 200 µm nozzle at 50-400 kPa. Also checks every Reynolds number
against the range they report (0.003 to 64).

    python scripts/validation/validate_flowrate_vs_ning_table1.py
"""
import csv
import os

import numpy as np

# Default to the canonical grid, the one every reported number is read from. Override with
# L8_PEN_CSV to point at a different sweep.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get(
    "L8_PEN_CSV",
    os.path.join(_REPO, "results", "fem", "exports", "L8_FEM_canonical_2026-08-27.csv"),
)
RHO = 990.0   # kg/m3, bioink density; converts the grid's uL/s to Ning's g/s

# Ning 2018 Table 1, L8 cell suspension, Ø200 µm needle. (mean, sd) in g/s.
NING_T1 = {
    50:  (2.6e-4, 1.0e-4),
    100: (1.8e-3, 0.46e-3),
    200: (1.7e-2, 0.58e-2),
    300: (4.1e-2, 0.77e-2),
    400: (9.8e-2, 0.95e-2),
}
NEEDLE = 200.0
NING_RE_MAX = 64.0


def main():
    with open(CSV) as f:
        rows = [{k: float(v) for k, v in r.items() if k != "source"}
                for r in csv.DictReader(f)]
    for r in rows:
        if "Q_gs" not in r:
            r["Q_gs"] = r["Q_uLs"] * 1e-9 * RHO * 1e3

    print("=" * 82)
    print("FEM FLOW RATE vs NING 2018 TABLE 1 (MEASURED)   needle Ø200 µm")
    print("=" * 82)
    print(f"{'P(kPa)':>7} {'measured(g/s)':>15} {'sd':>10} {'FEM(g/s)':>12} "
          f"{'FEM/exp':>8} {'within sd?':>11}")

    ratios, inside = [], 0
    for p, (mean, sd) in sorted(NING_T1.items()):
        m = [r for r in rows
             if abs(r["pressure_kPa"] - p) < 1e-6 and abs(r["needle_um"] - NEEDLE) < 1e-6]
        if not m:
            print(f"{p:7.0f} {mean:15.3e} {sd:10.2e} {'--- not in sweep ---':>12}")
            continue
        fem = m[0]["Q_gs"]
        ratio = fem / mean
        ok = abs(fem - mean) <= sd
        inside += bool(ok)
        ratios.append(ratio)
        print(f"{p:7.0f} {mean:15.3e} {sd:10.2e} {fem:12.3e} {ratio:8.2f} "
              f"{'YES' if ok else 'no':>11}")

    if ratios:
        r = np.array(ratios)
        print(f"\n  FEM/experiment: min {r.min():.2f}  max {r.max():.2f}  "
              f"median {np.median(r):.2f}")
        print(f"  within one experimental sd: {inside}/{len(ratios)} conditions")
        gm = float(np.exp(np.mean(np.log(r))))
        print(f"  geometric-mean bias: {gm:.2f}x  "
              f"({'FEM over-predicts' if gm > 1 else 'FEM under-predicts'})")

    print("\n" + "=" * 82)
    print("REYNOLDS NUMBER CROSS-CHECK")
    print("=" * 82)
    re_all = [r["Re"] for r in rows]
    print(f"  Ning report Re in this study spans 0.003 to {NING_RE_MAX:.0f}")
    print(f"  our sweep spans {min(re_all):.4f} to {max(re_all):.1f}")
    worst = max(rows, key=lambda r: r["Re"])
    print(f"  our maximum is at {worst['pressure_kPa']:.0f} kPa / "
          f"Ø{worst['needle_um']:.0f} µm")
    over = [r for r in rows if r["Re"] > NING_RE_MAX]
    print(f"  conditions exceeding Ning's stated maximum: {len(over)}/{len(rows)}")
    if over:
        print("    -> either we run beyond the regime Ning validated, or the two Re")
        print("       definitions differ.")

    # Same-pressure comparison Ning also ran: fixed 200 kPa, four diameters.
    print("\n" + "=" * 82)
    print("Ning et al. also measured flow rate at fixed 200 kPa for Ø100/200/250/330 µm")
    print("=" * 82)
    print("  (methods p13). Those values are not in Table 1; they appear in a figure.")
    for d in sorted({r["needle_um"] for r in rows}):
        m = [r for r in rows if r["needle_um"] == d and r["pressure_kPa"] == 200]
        if m:
            print(f"    our FEM at 200 kPa, Ø{d:3.0f} µm : {m[0]['Q_gs']:.4e} g/s")


if __name__ == "__main__":
    main()
