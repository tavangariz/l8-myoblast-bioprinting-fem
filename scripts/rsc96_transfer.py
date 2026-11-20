#!/usr/bin/env python3
"""The same pipeline for a second cell line and bioink (Additional file 1, Section S13, Table S16).

Ning et al. (2018) characterised two suspensions on the same cartridge and nozzles: L8 myoblasts and RSC96
Schwann cells, each in 1.5 % alginate, with their own power-law rheology and damage law. The RSC96 suspension
was solved on the 32-case grid (plus 450 and 500 kPa through the 100 µm nozzle, to reach the matched flow rate
of Ning et al.) by the same finite element model with K = 20.81, n = 0.37 (results/fem/runs/2026-10-31_rsc96).
Nothing below is fitted to RSC96 except D0, by the rule used for L8 (l8_calibration.py), on the RSC96
calibration series of Ning et al. Tested against what entered no fit:

  diameter series   damage measured at 200 kPa through 100, 250 and 330 µm nozzles (their Fig. 7A);
  matched flow      their comparison at about 3e-3 g/s across 100-330 µm nozzles (16.84 % at 100 µm,
                    7.28 % at 330 µm in their text);
  orderings         the damage ordering of the nozzles at fixed pressure and at matched flow rate.

The L8 control case of the same run (200 µm, 200 kPa, the L8 rheology through the same settings) must reproduce
the canonical flow rate, which shows the rheology setting changes nothing else. Writes
results/analyses/rsc96_transfer.json.

    python scripts/rsc96_transfer.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
from scipy.interpolate import PchipInterpolator

import l8_calibration as C
import ning2018_data as ning
from l8_paths import DATA

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "fem" / "runs" / "2026-10-31_rsc96"
DIAMS = (100.0, 200.0, 250.0, 330.0)
Q_CANONICAL_REF = 17.940002925163476


def uls(m_g_s):
    """Mass flow (g/s) to volume flow (uL/s) at the density Ning measured."""
    return m_g_s * 1e-3 / ning.DENSITY_KG_M3 * 1e9


def load(line):
    rows = []
    for f in sorted(RUN.glob(f"{line}_M12_D*_P*.csv")):
        r = next(csv.DictReader(open(f)))
        if int(float(r["converged"])) != 1 or float(r["rho_fraction"]) != 1.0:
            raise SystemExit(f"REFUSING: {f.name} did not converge at full density")
        rows.append(dict(D=float(r["needle_um"]), P=float(r["pressure_kPa"]), Q=float(r["Q_uLs"])))
    return rows


def main():
    ctrl = load("L8control")
    assert len(ctrl) == 1 and abs(ctrl[0]["Q"] / Q_CANONICAL_REF - 1) < 1e-6, ctrl
    rows = load("RSC96")
    for r in rows:
        r.update(C.chain(r["P"], r["Q"], r["D"] * 0.5e-6, "RSC96"))
    law = ning.LAW["RSC96"]
    by = {(r["D"], r["P"]): r for r in rows}
    series = ning.DAMAGE_PRESSURE_SERIES_200UM["RSC96"]
    cal = [by[(200.0, float(p))] for p in series]
    v = C.fit_variants(cal, np.array(list(series.values()), float), law)

    def dt(r, k):
        p = v[k]
        return 100 * float(C.total(r["tau_e_Pa"], r["Dst"], p["a2"], p["b2"], p["D0"]))

    out = dict(control_L8_Q_uLs=ctrl[0]["Q"], variants={k: {kk: vv for kk, vv in p.items()} for k, p in v.items()})
    # flow: measured vs solved at 200 um (Table 1) and 250/330 um at 200 kPa (Fig. 4C)
    flow = []
    for p, (m, s) in ning.FLOW_PRESSURE_SERIES_200UM_G_S["RSC96"].items():
        if (200.0, float(p)) in by:
            flow.append(dict(D=200.0, P=float(p), Q_fem=by[(200.0, float(p))]["Q"], Q_meas=uls(m)))
    for d, m in ning.FLOW_DIAMETER_SERIES_200KPA_G_S["RSC96"].items():
        flow.append(dict(D=float(d), P=200.0, Q_fem=by[(float(d), 200.0)]["Q"], Q_meas=uls(m)))
    for f in flow:
        f["offset_pct"] = round(100 * (f["Q_fem"] / f["Q_meas"] - 1), 1)
    out["flow"] = flow
    out["solved_Q_200kPa_uLs"] = {int(d): by[(d, 200.0)]["Q"] for d in DIAMS}
    # out-of-sample diameter series at 200 kPa
    oos = ning.DAMAGE_DIAMETER_SERIES_200KPA["RSC96"]
    out["diameter_series_200kPa"] = {k: [dict(D=d, pred=round(dt(by[(float(d), 200.0)], k), 2), meas=oos[d][0],
                                              bar=[oos[d][1], oos[d][2]],
                                              inside=bool(oos[d][1] <= dt(by[(float(d), 200.0)], k) <= oos[d][2]))
                                         for d in (100, 250, 330)] for k in v}
    # matched flow 3e-3 g/s and fixed pressure 200 kPa orderings
    q_t = uls(ning.MATCHED_FLOW_RSC96_G_S)
    out["matched_flow_uLs"] = q_t
    for k in v:
        m = {}
        for d in DIAMS:
            s = sorted([r for r in rows if r["D"] == d], key=lambda r: r["Q"])
            lq = np.log10([r["Q"] for r in s])
            if lq[0] <= np.log10(q_t) <= lq[-1]:
                m[d] = dict(P_kPa=round(float(10 ** PchipInterpolator(lq, np.log10([r["P"] for r in s]))(np.log10(q_t))), 1),
                            Dt=round(float(PchipInterpolator(lq, [dt(r, k) for r in s])(np.log10(q_t))), 2))
        out.setdefault("matched_flow", {})[k] = m
        # at the pressures Ning applied (their Fig. S4), PCHIP in log10 P along each diameter
        at_ning = {}
        for d, (p_n, meas) in ning.MATCHED_FLOW_RSC96_MEASURED.items():
            s = sorted([r for r in rows if r["D"] == float(d)], key=lambda r: r["P"])
            lp = np.log10([r["P"] for r in s])
            at_ning[d] = dict(P_kPa=p_n, measured=meas,
                              Q_uLs=round(float(10 ** PchipInterpolator(lp, np.log10([r["Q"] for r in s]))(np.log10(p_n))), 3),
                              Dt=round(float(PchipInterpolator(lp, [dt(r, k) for r in s])(np.log10(p_n))), 2))
        out.setdefault("at_ning_matched_pressures", {})[k] = at_ning
        out.setdefault("fixed_200kPa", {})[k] = {d: round(dt(by[(d, 200.0)], k), 2) for d in DIAMS}
    # the 1 % nozzle-friction test of the L8 main text, at Ning's matched-flow pressures (law O): the solved flow
    # of each nozzle at that pressure is held fixed and the friction subtracted in Eq. 17 is changed by 1 %
    po, fr = v["O"], {}
    for d, (p_n, meas) in ning.MATCHED_FLOW_RSC96_MEASURED.items():
        s = sorted([r for r in rows if r["D"] == float(d)], key=lambda r: r["P"])
        lp = np.log10([r["P"] for r in s])
        q = float(10 ** PchipInterpolator(lp, np.log10([r["Q"] for r in s]))(np.log10(p_n)))
        cell = {}
        for eps, key in ((-0.01, "minus1"), (0.0, "base"), (0.01, "plus1")):
            ch = C.chain(p_n, q, float(d) * 0.5e-6, "RSC96", fric_scale=1 + eps)
            cell[key] = round(100 * float(C.total(ch["tau_e_Pa"], ch["Dst"], po["a2"], po["b2"], po["D0"])), 2)
            cell[key + "_tau_e_Pa"] = round(ch["tau_e_Pa"], 0)
        assert abs(cell["base"] - out["at_ning_matched_pressures"]["O"][d]["Dt"]) < 0.1, (d, cell)
        fr[d] = cell
    ds_ = sorted(fr)
    fr["decrease_holds"] = {key: bool(all(fr[a][key] > fr[b][key] for a, b in zip(ds_, ds_[1:])))
                            for key in ("minus1", "base", "plus1")}
    out["friction_1pct_at_ning_matched_pressures_O"] = fr
    (DATA / "analyses").mkdir(parents=True, exist_ok=True)
    (DATA / "analyses" / "rsc96_transfer.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"L8 control Q = {ctrl[0]['Q']:.6f} uL/s (canonical {Q_CANONICAL_REF:.6f})")
    for k, p in v.items():
        print(f"[{k}] a2={p['a2']:.3e} b2={p['b2']:.3f} D0={100*p['D0']:.2f}% calib rms {p['fit_rms_pp']:.2f} pp"
              f" resid {p['fit_resid_pp']}")
        print(f"     diameter series 200 kPa: {out['diameter_series_200kPa'][k]}")
        print(f"     fixed 200 kPa: {out['fixed_200kPa'][k]} | matched {q_t:.2f} uL/s: {out['matched_flow'][k]}")
        print(f"     at Ning's matched-flow pressures: {out['at_ning_matched_pressures'][k]}")
    print("flow offsets:", [(f["D"], f["P"], f["offset_pct"]) for f in flow])
    print("friction 1% at Ning's matched-flow pressures (O):", out["friction_1pct_at_ning_matched_pressures_O"])
    for k in v:                                 # the claim: damage falls from 100 to 330 um at Ning's matched flow
        a = out["at_ning_matched_pressures"][k]
        steps = sum(a[x]["Dt"] > a[y]["Dt"] for x, y in ((100, 200), (200, 250), (250, 330)))
        print(f"[{k}] matched-flow decreasing steps: {steps} of 3 (measured: "
              f"{sum(a[x]['measured'] > a[y]['measured'] for x, y in ((100, 200), (200, 250), (250, 330)))} of 3)")


if __name__ == "__main__":
    main()
