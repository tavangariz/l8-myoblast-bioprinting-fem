"""Fix the FEM checks of the 20 % limit on the total damage before any of them is solved.

With the pressure-independent damage D0 of l8_calibration.py (variant O), the total predicted damage is
Dt = D0 + (1 - D0) Dn (Eq. 25), where Dn is the condition-dependent damage the surrogate emulates. Dt = 20 % is
therefore the contour Dn = (20 - 100 D0) / (1 - D0) of the surrogate, at lower pressures than the boundary the
eleven checks of prespecify_fem_checks.py tested. One case is placed on that contour at every trained and
intermediate diameter where it lies inside 25-400 kPa (200, 250, 290 and 330 µm). Writes the case list with the
predictions (PRESPECIFIED_CASES.csv) and the HTCondor submit file to results/fem/runs/2026-10-31_boundary-checks/;
none of the cases is used to fit or tune anything.

    python scripts/prespecify_boundary_checks.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

import l8_surrogate as S
from prespecify_fem_checks import SUB, MESH_M, TRAINED_D, INTERMEDIATE_D

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = "2026-10-31_boundary-checks"
OUT = ROOT / "results" / "fem" / "runs" / RUN
HEADER = ("# FEM checks of the 20 % limit on the total damage Dt = D0 + (1 - D0) Dn, fixed by\n"
          "# prespecify_boundary_checks.py before any case was solved. Same solver, image and settings as\n"
          "# the canonical grid; only the run directory and the case list differ.\n")


def main():
    d0 = json.loads((ROOT / "results" / "analyses" / "calibration_variants.json").read_text())["O"]["D0"]
    dn_limit = (S.DT_MAX - 100 * d0) / (1 - d0)
    gp_dt, gp_q, z = S.fit_all(check=True)
    P = np.linspace(*S.P_TRAINED, 3751)
    rows = []
    for d in TRAINED_D + INTERMEDIATE_D:
        dt = gp_dt.predict(z(P, np.full_like(P, d)))
        hit = np.flatnonzero(dt >= dn_limit)
        if hit.size == 0:
            print(f"  Ø{d:.0f}: Dn stays below {dn_limit:.2f} % up to {P[-1]:.0f} kPa")
            continue
        p = round(float(P[hit[0]]), 1)
        Xz = z([p], [d])
        m_dt, s_dt = (float(v[0]) for v in gp_dt.predict(Xz, return_std=True))
        m_q, s_q = (float(v[0]) for v in gp_q.predict(Xz, return_std=True))
        rows.append({"case": f"M{MESH_M}_D{d:g}_P{p:g}", "needle_um": d, "pressure_kPa": p,
                     "role": "dt_total_boundary", "gp_Dn_pct": round(m_dt, 4), "gp_Dn_sd_pp": round(s_dt, 4),
                     "gp_Dt_total_pct": round(100 * d0 + (1 - d0) * m_dt, 4),
                     "gp_log10_Q": round(m_q, 5), "gp_log10_Q_sd": round(s_q, 5),
                     "gp_Q_uLs": round(10 ** m_q, 5)})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "PRESPECIFIED_CASES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    queue = "\n".join(f"  {MESH_M}, {r['needle_um']:g}, {r['pressure_kPa']:g}" for r in rows)
    (OUT / "l8_boundary_checks.sub").write_text((HEADER + SUB.split("\n", 3)[3]).format(run=RUN, queue=queue))
    print(f"  D0 = {100 * d0:.3f} %, Dt = 20 % <=> Dn = {dn_limit:.3f} %")
    for r in rows:
        print(f"  {r['case']:<20} Dn {r['gp_Dn_pct']:6.2f} ± {r['gp_Dn_sd_pp']:.2f} pp  Dt {r['gp_Dt_total_pct']:6.2f} %"
              f"  Q {r['gp_Q_uLs']:9.3f} µL/s")


if __name__ == "__main__":
    main()
