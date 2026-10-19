#!/usr/bin/env python3
"""Table 1: wall shear stress and entrance pressure of the model against Ning et al. (2018), Table 2.

Prints the nine columns of Table 1 from the grid and the agreement quoted in the text, the flow
rates that their wall shear stresses imply through Eq. 16, and the extensional damage that their
extensional stresses give through Eq. 23.

    python scripts/table1_comparison.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8_damage as D  # noqa: E402
from l8_paths import CANONICAL_GRID, require  # noqa: E402

# Ning et al. (2018), Table 2, L8 cells: (nozzle diameter µm, applied pressure kPa) ->
# (contraction pressure drop Pa, wall shear stress Pa)
NING = {(200, 50): (867, 246), (200, 100): (1773, 491), (200, 200): (3734, 981),
        (200, 400): (18246, 1909), (100, 200): (1650, 496), (250, 200): (5620, 1214),
        (330, 200): (14345, 1532)}
# Ning et al. (2018), Table 2, L8 cells: extensional stress at 200 kPa (Pa)
NING_TAU_E = {200: 1893.0, 330: 7273.0}
ROWS = [(100, 200), (200, 50), (200, 100), (200, 200), (200, 300), (200, 400), (250, 200), (330, 200)]


def q_from_tau_w(tau_w: float, d_um: float) -> float:
    """Flow rate (µL/s) that gives wall shear stress tau_w in a nozzle of diameter d_um (Eq. 16 inverted)."""
    R = d_um / 2 * 1e-6
    n = D.N_EXP
    return math.pi * R ** 3 / 4 * (4 * n / (3 * n + 1)) * (tau_w / D.K_VIS) ** (1 / n) * 1e9


def main() -> int:
    grid = {(int(float(r["needle_um"])), float(r["pressure_kPa"])): r
            for r in csv.DictReader(open(require(CANONICAL_GRID, "the 32-case grid")))}
    print(f"{'D':>4} {'dP':>4} {'tau_w':>8} {'Ning':>6} {'d_tau%':>7} {'p_en':>8} {'Ning':>7} {'d_pen%':>7} {'Re':>8}")
    d_tau, d_pen, d_q = {}, {}, {}
    for d, p in ROWS:
        r = grid[(d, float(p))]
        tw, pen, q = float(r["tau_w_Pa"]), float(r["P_en_kPa"]), float(r["Q_uLs"])
        if (d, p) in NING:
            pen_n, tw_n = NING[(d, p)][0] / 1000, NING[(d, p)][1]
            d_tau[(d, p)] = 100 * (tw - tw_n) / tw_n
            d_pen[(d, p)] = 100 * (pen - pen_n) / pen_n
            d_q[(d, p)] = 100 * (q_from_tau_w(tw_n, d) - q) / q
            print(f"{d:>4} {p:>4} {tw:8.1f} {tw_n:6d} {d_tau[(d, p)]:+7.2f} {pen:8.3f} {pen_n:7.3f} "
                  f"{d_pen[(d, p)]:+7.2f} {float(r['Re']):8.3f}")
        else:
            print(f"{d:>4} {p:>4} {tw:8.1f} {'-':>6} {'-':>7} {pen:8.3f} {'-':>7} {'-':>7} {float(r['Re']):8.3f}")
    s200 = [k for k in d_tau if k[0] == 200]
    print(f"\nwall shear stress: within {max(abs(d_tau[k]) for k in s200):.2f} % for the 200 µm nozzle "
          f"(50-400 kPa), within {max(abs(v) for v in d_tau.values()):.2f} % over the seven conditions")
    low = [k for k in s200 if k[1] <= 200]
    at200 = sorted(abs(d_pen[k]) for k in d_pen if k[1] == 200)
    print(f"entrance pressure: within {max(abs(d_pen[k]) for k in low):.2f} % for the 200 µm nozzle at "
          f"50-200 kPa, {d_pen[(200, 400)]:+.2f} % at 400 kPa; {at200[0]:.2f} to {at200[-1]:.2f} % across "
          f"the four diameters at 200 kPa")
    print(f"flow rates implied by their wall shear stresses (Eq. 16): within "
          f"{max(abs(d_q[k]) for k in s200):.2f} % of the solved ones for the 200 µm nozzle (50-400 kPa), "
          f"{max(abs(v) for v in d_q.values()):.2f} % over the seven conditions")
    for d, te in NING_TAU_E.items():
        det_model = float(grid[(d, 200.0)]["Det_pct"])
        print(f"their extensional stress at {d} µm, 200 kPa ({te:.0f} Pa) gives Det = "
              f"{100 * D.compute_Det(te):.2f} % through Eq. 23; the model gives {det_model:.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
