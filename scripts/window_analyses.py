#!/usr/bin/env python3
"""What the operating window and the nozzle orderings depend on (Additional file 2, Section S9).

Four analyses from the 32 solved cases and the surrogate, with no new flow solution:

  calibration   the three readings of l8_calibration.py (P published, O published + D0, R refitted):
                window membership, Dt = 20 % boundaries, operating points, both diameter orderings;
  thresholds    the window and the nozzle choice under damage limits of 10-25 % and flow limits of
                0.1-10 µL/s (Table S7);
  extrapolation for every solved case, whether its stresses and residence time lie inside the ranges
                the two damage branches were fitted on, and by what factor the shear or the entrance
                term would have to change to move the case across the 20 % limit; the same factors
                for the matched-flow ordering (Table S8);
  flow          the damage chain re-evaluated at the flow rates Ning et al. measured instead of the solved
                ones (200 µm at 50-400 kPa, their Table 1; 250 and 330 µm at 200 kPa, their Fig. 4D), with
                the calibration held fixed and refitted (Table S9).

Variant O is the calibration of the article. Writes results/analyses/window_analyses.json.

    python scripts/window_analyses.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import PchipInterpolator

import l8_calibration as C
import l8_surrogate as S
import ning2018_data as ning
from l8_geometry import L_NEEDLE, TIP_RADIUS_M
from l8_paths import DATA

OUT = DATA / "analyses"
DIAMS = (100.0, 200.0, 250.0, 330.0)
P_GRID = np.linspace(*S.P_TRAINED, 3751)


def variant_dt(rows, v):
    """Total damage (%) of every grid case under calibration variant v."""
    return np.array([100 * float(C.total(r["tau_e_Pa"], r["Dst"], v["a2"], v["b2"], v["D0"])) for r in rows])


def surrogate_for(rows, dt):
    """The paper's surrogate configuration fitted to the given damage column."""
    grid = [dict(pressure_kPa=r["P"], needle_um=r["D"], Q_uLs=r["Q"], Dt_pct=float(d)) for r, d in zip(rows, dt)]
    return S.fit_all(grid, check=False)


def matched_flow(rows, dt, q_target):
    """Damage of each diameter at a matched flow rate, PCHIP in log10 Q over its solved pressures."""
    out = {}
    for d in DIAMS:
        idx = [i for i, r in enumerate(rows) if r["D"] == d]
        idx.sort(key=lambda i: rows[i]["Q"])
        lq = np.log10([rows[i]["Q"] for i in idx])
        if not lq[0] <= np.log10(q_target) <= lq[-1]:
            out[d] = None
            continue
        out[d] = float(PchipInterpolator(lq, dt[idx])(np.log10(q_target)))
    return out


def operating_point_dt(gp_dt, gp_q, z, d, q_min, dt_max):
    """S.operating_point for a surrogate fitted to the total damage of a variant, which carries that
    variant's own D0, so its prediction is Dt as it stands and takes no total_from_dn conversion."""
    Xz = z(P_GRID, np.full_like(P_GRID, d))
    dt, q = gp_dt.predict(Xz), 10 ** gp_q.predict(Xz)
    ok = (dt < dt_max) & (q >= q_min)
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[np.argmin(dt[ok])]
    return dict(P_kPa=float(P_GRID[i]), Dt_pct=float(dt[i]))


def window_summary(rows, dt, gp_dt, gp_q, z, dt_max=S.DT_MAX, q_min=S.Q_MIN):
    q = np.array([r["Q"] for r in rows])
    inside = (dt < dt_max) & (q >= q_min)
    ops, bounds = {}, {}
    for d in DIAMS + (150.0, 290.0):
        op = operating_point_dt(gp_dt, gp_q, z, d, q_min=q_min, dt_max=dt_max)
        ops[d] = None if op is None else dict(P_kPa=round(op["P_kPa"], 1), Dt_pct=round(op["Dt_pct"], 2))
        pred = gp_dt.predict(z(P_GRID, np.full_like(P_GRID, d)))
        hit = np.flatnonzero(pred >= dt_max)
        bounds[d] = None if hit.size == 0 else round(float(P_GRID[hit[0]]), 1)
    choice = min((d for d in DIAMS if ops[d]), key=lambda d: ops[d]["Dt_pct"], default=None)
    return dict(n_inside=int(inside.sum()),
                inside=[(r["D"], r["P"]) for r, k in zip(rows, inside) if k],
                operating_points=ops, dt20_boundary_kPa=bounds, lowest_damage_nozzle=choice)


def orderings(rows, dt):
    """Fixed-pressure rise with diameter (200-330 um, every P >= 75 kPa) and matched-flow inversion."""
    by = {(r["D"], r["P"]): d for r, d in zip(rows, dt)}
    fixed = all(by[(200.0, p)] < by[(250.0, p)] < by[(330.0, p)] for p in (75, 100, 150, 200, 300, 400))
    m1, m18 = matched_flow(rows, dt, 1.0), matched_flow(rows, dt, 17.94)
    inv1 = m1[100.0] > m1[200.0] > m1[250.0] > m1[330.0]
    inv18 = m18[200.0] > m18[250.0] > m18[330.0]
    return dict(fixed_pressure_rise_all_P_ge_75=fixed, at_200kPa={d: round(by[(d, 200.0)], 2) for d in DIAMS},
                rise_200_330_at_200kPa_pp=round(by[(330.0, 200.0)] - by[(200.0, 200.0)], 2),
                matched_1uLs={d: (None if v is None else round(v, 2)) for d, v in m1.items()},
                matched_17_94uLs={d: (None if v is None else round(v, 2)) for d, v in m18.items()},
                matched_inversion_1uLs=inv1, matched_inversion_17_94uLs=inv18)


def calibration_block(rows, variants):
    out = {}
    for k, v in variants.items():
        dt = variant_dt(rows, v)
        gp_dt, gp_q, z = surrogate_for(rows, dt)
        out[k] = dict(window=window_summary(rows, dt, gp_dt, gp_q, z), orderings=orderings(rows, dt),
                      Dt_range=[round(float(dt.min()), 2), round(float(dt.max()), 2)])
    return out


def threshold_block(rows, v):
    dt = variant_dt(rows, v)
    gp_dt, gp_q, z = surrogate_for(rows, dt)
    sweep = []
    for dmax in (10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0):
        for qmin in (0.1, 0.3, 1.0, 3.0, 10.0):
            w = window_summary(rows, dt, gp_dt, gp_q, z, dt_max=dmax, q_min=qmin)
            m = matched_flow(rows, dt, qmin)
            order = [d for d, _ in sorted(((d, x) for d, x in m.items() if x is not None), key=lambda t: t[1])]
            sweep.append(dict(dt_max=dmax, q_min=qmin, n_inside=w["n_inside"],
                              lowest_damage_nozzle=w["lowest_damage_nozzle"],
                              matched_flow_order_low_to_high=order,
                              operating_points={d: w["operating_points"][d] for d in DIAMS}))
    return sweep


def extrapolation_block(rows, v, dt_max=S.DT_MAX):
    """Calibration-range flags and the factor on each damage term that moves a case across the limit."""
    ext_lo, ext_hi = 440.0, 9251.0
    out = []
    for r in rows:
        det = float(C.det(r["tau_e_Pa"], v["a2"], v["b2"]))
        dst, d0 = r["Dst"], v["D0"]
        dt = 1 - (1 - d0) * (1 - det) * (1 - dst)
        target = dt_max / 100

        def factor(term):          # multiply one term (det or dst) by f so that Dt == limit
            other = dst if term == "det" else det
            need = 1 - (1 - target) / ((1 - d0) * (1 - other))
            base = det if term == "det" else dst
            return None if base <= 0 or need <= 0 else round(need / base, 3)
        res_t = L_NEEDLE / (r["Q"] * 1e-9 / (np.pi * (r["D"] * 0.5e-6) ** 2))
        out.append(dict(D=r["D"], P=r["P"], Q_uLs=round(r["Q"], 4), Dt_pct=round(100 * dt, 3),
                        tau_e_Pa=round(r["tau_e_Pa"], 1), tau_e_in_range=ext_lo <= r["tau_e_Pa"] <= ext_hi,
                        tau_w_Pa=round(r["tau_w_Pa"], 1), tau_w_in_range=100 <= r["tau_w_Pa"] <= 1000,
                        mean_residence_s=round(float(res_t), 4),
                        inside=bool(100 * dt < dt_max and r["Q"] >= S.Q_MIN),
                        factor_on_Det_to_reach_limit=factor("det"), factor_on_Dst_to_reach_limit=factor("dst")))
    return out


def matched_flow_margins(rows, v, q_target=1.0):
    """Damage at a matched flow rate when the shear term of every case is multiplied by f, the test
    of the exposure-time extrapolation of the shear branch; and the entrance stress at that flow."""
    out = {}
    for f in (1.0, 2.0, 3.0, 5.0, 10.0):
        vv = dict(v)
        dt = np.array([100 * float(1 - (1 - vv["D0"]) * (1 - C.det(r["tau_e_Pa"], vv["a2"], vv["b2"]))
                                   * (1 - min(f * r["Dst"], 1.0))) for r in rows])
        m = matched_flow(rows, dt, q_target)
        out[f"Dst_x{f:g}"] = {d: (None if x is None else round(x, 2)) for d, x in m.items()}
    taue = {}
    for d in DIAMS:
        idx = sorted([i for i, r in enumerate(rows) if r["D"] == d], key=lambda i: rows[i]["Q"])
        lq = np.log10([rows[i]["Q"] for i in idx])
        if lq[0] <= np.log10(q_target) <= lq[-1]:
            taue[d] = round(float(PchipInterpolator(lq, [rows[i]["tau_e_Pa"] for i in idx])(np.log10(q_target))), 1)
    out["tau_e_at_target_Pa"] = taue
    return out


def flow_block(rows, variants):
    """The chain at Ning's measured flow rates (mass flow / 0.99 g/mL)."""
    by = {(r["D"], r["P"]): r for r in rows}
    meas = [(200.0, float(p), m, s) for p, (m, s) in ning.FLOW_PRESSURE_SERIES_200UM_G_S["L8"].items()]
    meas += [(float(d), 200.0, m, None) for d, m in ning.FLOW_DIAMETER_SERIES_200KPA_G_S["L8"].items()]
    O = variants["O"]
    out = []
    for d, p, m, s in meas:
        q_meas = m / (ning.DENSITY_KG_M3 * 1e-3) * 1e3       # g/s -> uL/s
        fem = by[(d, p)]
        row = dict(D=d, P=p, Q_fem_uLs=round(fem["Q"], 3), Q_meas_uLs=round(q_meas, 3),
                   Q_offset_pct=round(100 * (fem["Q"] / q_meas - 1), 1))
        for tag, q in (("fem", fem["Q"]), ("meas", q_meas)) + ((("meas_lo", (m - s) / 0.99e-3 * 1e-3 * 1e3),
                                                                 ("meas_hi", (m + s) / 0.99e-3 * 1e-3 * 1e3)) if s else ()):
            c = C.chain(p, q, d * 0.5e-6, "L8")
            row[f"Dt_O_{tag}"] = round(100 * float(C.total(c["tau_e_Pa"], c["Dst"], O["a2"], O["b2"], O["D0"])), 2)
        out.append(row)
    # recalibrate D0 on the chain at measured flows (200 um series), so flow and calibration are separated
    series = ning.DAMAGE_PRESSURE_SERIES_200UM["L8"]
    cal_meas = []
    for p in series:
        m, _ = ning.FLOW_PRESSURE_SERIES_200UM_G_S["L8"][p]
        cal_meas.append(C.chain(float(p), m / 0.99e-3 * 1e-3 * 1e3, TIP_RADIUS_M[200], "L8"))
    v_meas = C.fit_variants(cal_meas, np.array(list(series.values()), float), ning.LAW["L8"])
    # operating pressure of the 200 um nozzle at 1 uL/s, solved versus measured flow (log-log interpolation)
    pm = sorted(ning.FLOW_PRESSURE_SERIES_200UM_G_S["L8"].items())
    p_meas = float(np.exp(np.interp(0.0, np.log([m / 0.99e-3 * 1e-3 * 1e3 for _, (m, _) in pm]),
                                    np.log([p for p, _ in pm]))))
    idx = sorted([r for r in rows if r["D"] == 200.0], key=lambda r: r["Q"])
    p_fem = float(np.exp(np.interp(0.0, np.log([r["Q"] for r in idx]), np.log([r["P"] for r in idx]))))
    return dict(points=out, D0_with_measured_flow=round(100 * v_meas["O"]["D0"], 3),
                O_rms_with_measured_flow=round(v_meas["O"]["fit_rms_pp"], 3),
                op_pressure_200um_1uLs_kPa=dict(fem=round(p_fem, 1), measured=round(p_meas, 1)))


def main():
    rows = C.grid_rows("L8")
    variants = json.loads((OUT / "calibration_variants.json").read_text())
    res = dict(calibration=calibration_block(rows, variants),
               thresholds=threshold_block(rows, variants["O"]),
               extrapolation=extrapolation_block(rows, variants["O"]),
               matched_flow_margins=matched_flow_margins(rows, variants["O"]),
               flow=flow_block(rows, variants))
    (OUT / "window_analyses.json").write_text(json.dumps(res, indent=1, default=str))
    for k, b in res["calibration"].items():
        w, o = b["window"], b["orderings"]
        print(f"[{k}] inside {w['n_inside']}/32 | boundary {w['dt20_boundary_kPa']} | ops {w['operating_points']}")
        print(f"     200 kPa {o['at_200kPa']} rise {o['rise_200_330_at_200kPa_pp']} pp | fixed-P rise all P>=75: "
              f"{o['fixed_pressure_rise_all_P_ge_75']} | 1 uL/s {o['matched_1uLs']} inv {o['matched_inversion_1uLs']}"
              f" | 17.94 {o['matched_17_94uLs']} inv {o['matched_inversion_17_94uLs']} | range {b['Dt_range']}")


if __name__ == "__main__":
    main()
