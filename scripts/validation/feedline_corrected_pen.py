#!/usr/bin/env python3
"""Entrance pressure with the wall friction of the cartridge and transition removed.

The entrance pressure is the residual of the applied pressure and the developed nozzle drop, so
it includes the wall friction upstream of the nozzle. This script subtracts that friction
(fully developed power-law flow in the cartridge and the transition) and compares both versions
with the contraction pressure drops of Ning et al. (2018), Table 2. Subtracting it moves every
value away from theirs, which is why the full residual is used.

    python scripts/validation/feedline_corrected_pen.py
"""
import csv
import math
import os
import sys

K_VIS, N_EXP = 25.60, 0.35

# Feed-line geometry of this screening test: three sequential straight sections, 32.5 mm
# in all. The published grid is solved on the Ning domain of l8_geometry (a 10.5 mm straight
# cartridge, a 1.5 mm funnel, then the tip), so this test does not correct the published grid.
R_BARREL, L_BARREL = 2.1e-3, 12.0e-3      # z 0 -> 12 mm
R_MID, L_MID = 1.0e-3, 10.5e-3            # z 12 -> 22.5 mm
L_TIP = 10.0e-3                           # z 22.5 -> 32.5 mm

# Ning et al. 2018 Table 2: (P_kPa, D_um) -> P_en in Pa, from their COMSOL model,
# itself validated against their weighed mass flow rates.
NING_T2 = {
    (50, 200): 867, (100, 200): 1773, (200, 200): 3734, (400, 200): 18246,
    (200, 100): 1650, (200, 250): 5620, (200, 330): 14345,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.environ.get(
    "L8_CANONICAL_CSV",
    os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                 "results", "fem", "exports", "L8_FEM_canonical_2026-08-27.csv"))


def developed_drop(Q_m3s, R, L):
    """Fully-developed power-law pressure drop along a straight tube.

    Rabinowitsch wall shear rate for a power-law fluid, then tau_w = K*gdot^n, then
    the force balance dP = 2*L*tau_w/R.
    """
    if Q_m3s <= 0:
        return 0.0
    gdot_app = 4.0 * Q_m3s / (math.pi * R ** 3)
    gdot_w = (3.0 * N_EXP + 1.0) / (4.0 * N_EXP) * gdot_app
    tau_w = K_VIS * gdot_w ** N_EXP
    return 2.0 * L * tau_w / R


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"canonical grid not found: {CSV_PATH}")
    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"canonical grid: {CSV_PATH}  ({len(rows)} rows)\n")

    col_tau = "tau_w_Pa" if "tau_w_Pa" in rows[0] else "tau_wall_Pa"

    print("FEED-LINE SHARE OF THE RESIDUAL, ALL CONDITIONS")
    print(f"  {'D':>4} {'P':>5} {'Q(uL/s)':>10} {'P_en res':>10} {'barrel':>8} "
          f"{'mid':>8} {'feed %':>7} {'corrected':>10}")
    shares, corrected = [], {}
    for r in rows:
        D = float(r["needle_um"])
        P = float(r["pressure_kPa"])
        Q = float(r["Q_uLs"]) * 1e-9
        pen = float(r["P_en_kPa"]) * 1e3
        db = developed_drop(Q, R_BARREL, L_BARREL)
        dm = developed_drop(Q, R_MID, L_MID)
        feed = db + dm
        share = 100.0 * feed / pen if pen > 0 else float("nan")
        shares.append(share)
        corrected[(int(P), int(D))] = pen - feed
        print(f"  {D:4.0f} {P:5.0f} {float(r['Q_uLs']):10.3f} {pen:10.1f} "
              f"{db:8.1f} {dm:8.1f} {share:6.1f}% {pen - feed:10.1f}")

    print(f"\n  feed-line share spans {min(shares):.1f}% to {max(shares):.1f}% "
          f"of the residual\n")

    print("SCORED AGAINST NING 2018 TABLE 2  (ratio ours/theirs, 1.000 is perfect)")
    print(f"  {'D':>4} {'P':>5} {'Ning':>9} {'residual':>10} {'ratio':>7} "
          f"{'corrected':>10} {'ratio':>7}   verdict")
    res_err, cor_err = [], []
    for (P, D), ning in sorted(NING_T2.items()):
        row = [r for r in rows
               if abs(float(r["needle_um"]) - D) < 1e-6
               and abs(float(r["pressure_kPa"]) - P) < 1e-6]
        if not row:
            print(f"  {D:4d} {P:5d} {ning:9d}   (not in this grid)")
            continue
        pen = float(row[0]["P_en_kPa"]) * 1e3
        cor = corrected[(P, D)]
        rr, cr = pen / ning, cor / ning
        res_err.append(abs(math.log(rr)))
        cor_err.append(abs(math.log(cr)))
        better = "closer" if abs(math.log(cr)) < abs(math.log(rr)) else "WORSE"
        print(f"  {D:4d} {P:5d} {ning:9d} {pen:10.1f} {rr:7.3f} "
              f"{cor:10.1f} {cr:7.3f}   {better}")

    if res_err:
        print(f"\n  mean |log ratio|   residual {sum(res_err)/len(res_err):.4f}"
              f"   corrected {sum(cor_err)/len(cor_err):.4f}")
        print(f"  geometric-mean ratio to Ning:")
        print(f"    residual  {math.exp(sum(math.log(float(r['P_en_kPa'])*1e3/NING_T2[(int(float(r['pressure_kPa'])), int(float(r['needle_um'])))]) for r in rows if (int(float(r['pressure_kPa'])), int(float(r['needle_um']))) in NING_T2) / len([r for r in rows if (int(float(r['pressure_kPa'])), int(float(r['needle_um']))) in NING_T2])):.3f}")
        print(f"    corrected {math.exp(sum(math.log(corrected[k]/NING_T2[k]) for k in corrected if k in NING_T2) / len([k for k in corrected if k in NING_T2])):.3f}")
        verdict = ("SUBTRACTING THE FEED LINE HELPS"
                   if sum(cor_err) < sum(res_err)
                   else "SUBTRACTING THE FEED LINE DOES NOT HELP")
        print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
