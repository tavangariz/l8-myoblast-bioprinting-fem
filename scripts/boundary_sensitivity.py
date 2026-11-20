#!/usr/bin/env python3
"""The Dt = 20 % boundary under one standard error of D0, and the operating points at the measured flow.

d0_uncertainty_boundary.json
    D0 of calibration variant O (l8_calibration.py) moved by one standard error each way. The 20 % limit on the
    total damage becomes the limit Dn = (20 - 100 D0) / (1 - D0) on the surrogate, and the boundary at 200,
    250, 290 and 330 µm is the lowest pressure on a 0.1 kPa grid (25-400 kPa) where the surrogate reaches it.
    The shifts are taken at the three solved diameters with a boundary inside the grid (Additional file 1,
    Table S10).

measured_flow_operating_points.json
    The predicted damage at each operating point (results/optimal_params/L8_operating_points.csv) and at the
    pressure where the measured flow reaches the same 1 µL/s. For 200 µm that pressure is read from the 200 µm
    flow series of Ning et al. (window_analyses.json, flow block); for 250 and 330 µm the operating pressure is
    scaled by the flow offset measured at 200 kPa, P' = P (1 + offset)^n, since Q grows as P^(1/n) along one
    nozzle; 100 µm has no measured flow.

Both files are written to results/analyses/.

    python scripts/boundary_sensitivity.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import l8_surrogate as S
from l8_damage import N_EXP

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "analyses"


def d0_boundary(gp, z) -> dict:
    cal = json.loads((REV / "calibration_variants.json").read_text())["O"]
    d0, se = cal["D0"], cal["se"]["D0"]
    P = np.linspace(25, 400, 3751)
    out = {}
    for lab, D in (("D0-se", d0 - se), ("D0", d0), ("D0+se", d0 + se)):
        lim = (20 - 100 * D) / (1 - D)
        b = {}
        for d in (200.0, 250.0, 290.0, 330.0):
            dn = gp.predict(z(P, np.full_like(P, d)))
            k = np.flatnonzero(dn >= lim)
            b[d] = None if k.size == 0 else round(float(P[k[0]]), 1)
        out[lab] = dict(D0_pct=round(100 * D, 2), Dn_limit=round(lim, 2), boundary=b)
    sh = [abs(out[x]["boundary"][d] - out["D0"]["boundary"][d]) for x in ("D0-se", "D0+se") for d in (200.0, 250.0, 330.0)]
    out["max_shift_kPa"] = round(max(sh), 1)
    out["min_shift_kPa"] = round(min(sh), 1)
    return out


def measured_flow_points(gp, z) -> dict:
    w = json.loads((REV / "window_analyses.json").read_text())
    fl = {(p["D"], p["P"]): p for p in w["flow"]["points"]}
    with open(ROOT / "results" / "optimal_params" / "L8_operating_points.csv") as fh:
        ops = {int(float(r["needle_um"])): float(r["P_kPa"]) for r in csv.DictReader(fh)}

    def dt(p, d):
        return float(S.total_from_dn(gp.predict(z([p], [float(d)])))[0])

    out = {}
    for d, P in ops.items():
        if d == 200:
            Pm = w["flow"]["op_pressure_200um_1uLs_kPa"]["measured"]
        elif (float(d), 200.0) in fl:
            Pm = P * (1 + fl[(float(d), 200.0)]["Q_offset_pct"] / 100) ** N_EXP
        else:
            Pm = None
        out[d] = dict(P_model=P, Dt_model=round(dt(P, d), 2), P_meas=None if Pm is None else round(Pm, 1),
                      Dt_meas=None if Pm is None else round(dt(Pm, d), 2))
    return out


if __name__ == "__main__":
    gp_dt, _gp_q, z = S.fit_all(check=False)
    b = d0_boundary(gp_dt, z)
    json.dump(b, open(REV / "d0_uncertainty_boundary.json", "w"), indent=1)
    print(f"D0 +/- 1 s.e.: boundary moves {b['min_shift_kPa']}-{b['max_shift_kPa']} kPa")
    m = measured_flow_points(gp_dt, z)
    json.dump(m, open(REV / "measured_flow_operating_points.json", "w"), indent=1)
    for d, r in m.items():
        print(f"  {d} um: P {r['P_model']} kPa Dt {r['Dt_model']} %   at measured flow: P {r['P_meas']} Dt {r['Dt_meas']}")
