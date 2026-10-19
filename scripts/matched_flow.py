#!/usr/bin/env python3
"""Total damage of each nozzle at a matched flow rate (Results, Diameter ordering at matched flow rate).

The grid is a pressure sweep, so for each nozzle the total damage and the applied pressure at a
given flow rate are interpolated monotonically (PCHIP) in log10 Q; linear interpolation in log10 Q
is printed beside it. Default flow rates: 1 µL/s and the flow rate of the reference condition
(200 kPa, 200 µm nozzle).

    python scripts/matched_flow.py [--q 1 17.94]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l8_paths import CANONICAL_GRID, require  # noqa: E402


def main() -> int:
    rows = list(csv.DictReader(open(require(CANONICAL_GRID, "the 32-case grid"))))
    by = {}
    for r in rows:
        by.setdefault(int(float(r["needle_um"])), []).append(
            (float(r["pressure_kPa"]), float(r["Q_uLs"]), float(r["Dt_pct"])))
    q_ref = next(q for p, q, _ in by[200] if p == 200.0)
    ap = argparse.ArgumentParser(description="Total damage of each nozzle at a matched flow rate.")
    ap.add_argument("--q", type=float, nargs="+", default=[1.0, q_ref], help="flow rates, µL/s")
    args = ap.parse_args()
    for q_t in args.q:
        print(f"\nflow rate {q_t:.2f} µL/s")
        print(f"  {'nozzle':>8} {'pressure (kPa)':>15} {'Dt (%)':>8} {'Dt, linear in log Q':>20}")
        for d in sorted(by):
            p, q, dt = (np.array(v) for v in zip(*sorted(by[d])))
            if not q.min() <= q_t <= q.max():
                print(f"  {d:>5} µm   flow rate not reached in 25-400 kPa (largest {q.max():.2f} µL/s)")
                continue
            x, lq = np.log10(q_t), np.log10(q)
            print(f"  {d:>5} µm {float(PchipInterpolator(lq, p)(x)):15.1f} "
                  f"{float(PchipInterpolator(lq, dt)(x)):8.2f} {float(np.interp(x, lq, dt)):20.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
