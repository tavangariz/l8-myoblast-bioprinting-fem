#!/usr/bin/env python3
"""Compare the surrogate with the four FEM checks of the 20 % limit on the total damage (Additional file 1, Table S5).

The cases were fixed by prespecify_boundary_checks.py before any was solved. Errors are on Dn, the quantity
the surrogate emulates; window membership is decided on the total Dt = D0 + (1 - D0) Dn. Writes
checks_vs_surrogate.csv in the run folder, in the columns of the eleven checks of evaluate_fem_checks.py, plus
the surrogate's total and the pressure by which the FEM moves the limit: (20 % - FEM total) over the slope of
the surrogate's total in pressure at the check (central difference over +/- 1 kPa).

    python scripts/evaluate_boundary_checks.py
"""
from __future__ import annotations

import csv
import math
import pathlib

import numpy as np

import l8_surrogate as S
from l8_damage import derive

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "fem" / "runs" / "2026-10-31_boundary-checks"
Z95 = 1.959963984540054


def slope(gp, z, d, p):
    P = np.array([p - 1.0, p + 1.0])
    t = S.total_from_dn(gp.predict(z(P, np.full(2, d))))
    return (t[1] - t[0]) / 2.0


def main():
    gp, _, z = S.fit_all(S.load_grid(), check=True)
    out = []
    for c in csv.DictReader(open(RUN / "PRESPECIFIED_CASES.csv")):
        r = next(csv.DictReader(open(RUN / f"{c['case']}.csv")))
        if int(float(r["converged"])) != 1 or float(r["rho_fraction"]) != 1.0:
            raise SystemExit(f"REFUSING: {c['case']} did not converge at full density")
        d, p, q = float(c["needle_um"]), float(c["pressure_kPa"]), float(r["Q_uLs"])
        fem = derive(p, q, d * 0.5e-6)
        dn_pred, sd = float(c["gp_Dn_pct"]), float(c["gp_Dn_sd_pp"])
        q_pred, lq_sd = float(c["gp_Q_uLs"]), float(c["gp_log10_Q_sd"])
        out.append({"case": c["case"], "role": c["role"], "needle_um": d, "pressure_kPa": p,
                    "Dt_fem_pct": round(fem["Dn_pct"], 4), "Dt_pred_pct": dn_pred,
                    "Dt_err_pp": round(dn_pred - fem["Dn_pct"], 4),
                    "Dt_inside95": int(abs(dn_pred - fem["Dn_pct"]) <= Z95 * sd),
                    "Q_fem_uLs": round(q, 5), "Q_pred_uLs": q_pred,
                    "Q_err_pct": round((q_pred / q - 1) * 100, 3),
                    "log10_Q_inside95": int(abs(math.log10(q_pred) - math.log10(q)) <= Z95 * lq_sd),
                    "Dt_total_fem_pct": round(fem["Dt_pct"], 4),
                    "accepted_fem": int(fem["Dt_pct"] < S.DT_MAX and q >= S.Q_MIN),
                    "accepted_pred": int(float(S.total_from_dn(dn_pred)) < S.DT_MAX and q_pred >= S.Q_MIN),
                    "Dt_total_pred_pct": round(float(S.total_from_dn(dn_pred)), 4),
                    "limit_offset_kPa": round(float((S.DT_MAX - fem["Dt_pct"]) / slope(gp, z, d, p)), 2)})
    with open(RUN / "checks_vs_surrogate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


if __name__ == "__main__":
    main()
