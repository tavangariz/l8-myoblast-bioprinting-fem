#!/usr/bin/env python3
"""Compare the surrogate with the eleven FEM checks of the operating window (Additional file 1, Table S5).

The checks were fixed by prespecify_fem_checks.py from the surrogate's predictions before any of
them was solved. Each solved check is reduced through l8_damage.derive, as a grid case, and
compared with the prediction recorded in PRESPECIFIED_CASES.csv; nothing is refitted. The script
stops unless every check is present and converged at full density.

    python scripts/evaluate_fem_checks.py
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys

from l8_damage import derive
import l8_surrogate as S

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "fem" / "runs" / \
    "2026-09-23_independent-checks"
Z95 = 1.959963985


def main() -> int:
    cases = list(csv.DictReader(open(RUN / "PRESPECIFIED_CASES.csv")))
    out, missing, failed = [], [], []
    for c in cases:
        f = RUN / f"{c['case']}.csv"
        if not f.exists():
            missing.append(c["case"])
            continue
        r = next(csv.DictReader(open(f)))
        if int(float(r["converged"])) != 1 or float(r["rho_fraction"]) != 1.0:
            failed.append(c["case"])
            continue
        d, p, q = float(c["needle_um"]), float(c["pressure_kPa"]), float(r["Q_uLs"])
        fem = derive(p, q, d * 0.5e-6)
        dt_pred, dt_sd = float(c["gp_Dt_pct"]), float(c["gp_Dt_sd_pp"])
        lq_pred, lq_sd = float(c["gp_log10_Q"]), float(c["gp_log10_Q_sd"])
        ok_fem = fem["Dt_pct"] < S.DT_MAX and q >= S.Q_MIN
        ok_pred = dt_pred < S.DT_MAX and 10 ** lq_pred >= S.Q_MIN
        out.append({"case": c["case"], "role": c["role"], "needle_um": d, "pressure_kPa": p,
                    "Dt_fem_pct": round(fem["Dt_pct"], 4), "Dt_pred_pct": dt_pred,
                    "Dt_err_pp": round(dt_pred - fem["Dt_pct"], 4),
                    "Dt_inside95": int(abs(dt_pred - fem["Dt_pct"]) <= Z95 * dt_sd),
                    "Q_fem_uLs": round(q, 5), "Q_pred_uLs": round(10 ** lq_pred, 5),
                    "Q_err_pct": round((10 ** lq_pred / q - 1) * 100, 3),
                    "log10_Q_inside95": int(abs(lq_pred - math.log10(q)) <= Z95 * lq_sd),
                    "accepted_fem": int(ok_fem), "accepted_pred": int(ok_pred)})
    if missing or failed:
        print(f"  MISSING {len(missing)}: {', '.join(missing) or '-'}")
        print(f"  NOT AT FULL DENSITY {len(failed)}: {', '.join(failed) or '-'}")
        print("  No verdict: every prespecified case must be solved and converged.")
        return 1

    with open(RUN / "checks_vs_surrogate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(f"  {o['case']:<18} {o['role']:<12} D_t FEM {o['Dt_fem_pct']:7.3f}  GP "
              f"{o['Dt_pred_pct']:7.3f}  err {o['Dt_err_pp']:+.3f} pp {'in ' if o['Dt_inside95'] else 'OUT'}"
              f" | Q FEM {o['Q_fem_uLs']:9.3f}  GP {o['Q_pred_uLs']:9.3f}  err {o['Q_err_pct']:+7.2f} %"
              f" {'in ' if o['log10_Q_inside95'] else 'OUT'} | accept FEM {o['accepted_fem']} GP "
              f"{o['accepted_pred']}")
    n = len(out)
    mae = sum(abs(o["Dt_err_pp"]) for o in out) / n
    worst = max(abs(o["Dt_err_pp"]) for o in out)
    qmape = sum(abs(o["Q_err_pct"]) for o in out) / n
    wrong = sum(o["accepted_fem"] != o["accepted_pred"] for o in out)
    print(f"\n  {n} checks: D_t MAE {mae:.3f} pp, worst {worst:.3f} pp; Q mean |error| {qmape:.2f} %;"
          f" D_t inside 95 % {sum(o['Dt_inside95'] for o in out)}/{n}, Q inside 95 % "
          f"{sum(o['log10_Q_inside95'] for o in out)}/{n}; decisions that differ {wrong}/{n}")
    print(f"  written {(RUN / 'checks_vs_surrogate.csv').relative_to(RUN.parents[3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
