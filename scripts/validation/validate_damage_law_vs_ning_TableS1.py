#!/usr/bin/env python3
"""Shear-damage law against the 20 measurements of Ning et al. (2018), Supporting Information, Table S1, to which a1, b1 and c1 were fitted.

    python scripts/validation/validate_damage_law_vs_ning_TableS1.py
"""
import csv
import os

import numpy as np

A1, B1, C1 = 3.20e-4, 0.61, 0.52          # Ning 2018, L8
NING_R2_REPORTED = 0.91

# Ning 2018 Supporting Information, Table S1, L8 block. Rows = shear time (s),
# columns = shear stress (Pa). Transcribed verbatim from the SI PDF.
TAU = [100, 400, 600, 800, 1000]
TIME = [10, 30, 60, 120]
EXP = {
    10:  [2.11, 3.22, 5.42, 6.18, 8.80],
    30:  [4.73, 5.13, 8.60, 8.29, 12.25],
    60:  [6.30, 8.62, 11.64, 12.08, 18.05],
    120: [9.06, 13.23, 15.44, 17.26, 25.49],
}

# The two extended-range checks Ning ran separately (paper p19), used as held-out points.
HELDOUT = [(900.0, 10.0, 5.29), (1700.0, 100.0, 29.55)]

# Repo-relative by default so the script runs anywhere; override with L8_VALID_OUT.
OUT = os.environ.get(
    "L8_VALID_OUT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results", "validation",
        "damage_law_validation_vs_NingTableS1.csv"),
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def D_s(tau, t):
    return (1.0 - np.exp(-A1 * np.asarray(tau, float) ** B1
                         * np.asarray(t, float) ** C1)) * 100.0


def main():
    rows, obs, pred = [], [], []
    for t in TIME:
        for j, tau in enumerate(TAU):
            e = EXP[t][j]
            p = float(D_s(tau, t))
            rows.append(dict(tau_Pa=tau, time_s=t, exp_pct=e, model_pct=round(p, 3),
                             abs_err_pp=round(p - e, 3),
                             rel_err_pct=round(100 * (p - e) / e, 2), set="TableS1"))
            obs.append(e)
            pred.append(p)

    obs_a, pred_a = np.array(obs), np.array(pred)
    resid = pred_a - obs_a
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs_a - obs_a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))

    print(f"{'tau(Pa)':>8} {'t(s)':>5} {'exp%':>7} {'model%':>8} {'err(pp)':>8} {'err(%)':>8}")
    for r in rows:
        print(f"{r['tau_Pa']:8d} {r['time_s']:5d} {r['exp_pct']:7.2f} "
              f"{r['model_pct']:8.2f} {r['abs_err_pp']:+8.2f} {r['rel_err_pct']:+8.1f}")

    print(f"\nn = {len(obs)} experimental points (Ning 2018 Table S1, L8)")
    print(f"R^2   = {r2:.4f}   (Ning report {NING_R2_REPORTED} for their L8 fit)")
    print(f"RMSE  = {rmse:.3f} percentage points")
    print(f"MAE   = {mae:.3f} pp | mean bias = {bias:+.3f} pp")

    print("\nHeld-out extended-range checks (Ning 2018 p19):")
    for tau, t, e in HELDOUT:
        p = float(D_s(tau, t))
        rows.append(dict(tau_Pa=tau, time_s=t, exp_pct=e, model_pct=round(p, 3),
                         abs_err_pp=round(p - e, 3),
                         rel_err_pct=round(100 * (p - e) / e, 2), set="heldout_p19"))
        print(f"  tau={tau:6.0f} Pa, t={t:5.0f} s : exp {e:5.2f}%  model {p:5.2f}%  "
              f"({p - e:+.2f} pp)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")

    if r2 < 0.80:
        raise SystemExit(
            f"R^2 = {r2:.3f} is far below Ning's reported {NING_R2_REPORTED}. The "
            "implementation does not reproduce the source fit — investigate before using."
        )


if __name__ == "__main__":
    main()
