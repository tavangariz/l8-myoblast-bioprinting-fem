#!/usr/bin/env python3
"""Numbers the article and Additional file 1 quote that no other script writes, each computed here.

matched-flow stresses  tau_e at 1 and 17.94 µL/s for every nozzle (PCHIP in log10 Q), and the
                       monotone bound on the 330 µm point that lies below the 440 Pa floor;
upper boundary         tau_e where Dt = 20 % is reached at 200, 250 and 330 µm;
surrogate settings     the Dt = 20 % boundary and window membership near it under the kernel and
                       regularization variants of Additional file 1, Table S14;
measured-flow point    Dt at the pressure where the measured 200 µm flow reaches 1 µL/s;
nozzle friction        the Dt = 20 % boundary with the developed nozzle friction scaled by 1 % each way
                       (Additional file 1, Table S11).

Writes results/analyses/extra_numbers.json.

    python scripts/extra_numbers.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import l8_calibration as C
import l8_surrogate as S
from l8_damage import D0, derive
from l8_paths import DATA

DIAMS = (100.0, 200.0, 250.0, 330.0)


def main():
    rows = S.load_grid()
    out = {}
    # 1. tau_e at matched flow
    for qt in (1.0, 17.94):
        te = {}
        for d in DIAMS:
            s = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["Q_uLs"])
            lq = np.log10([r["Q_uLs"] for r in s])
            if lq[0] <= np.log10(qt) <= lq[-1]:
                te[d] = round(float(PchipInterpolator(lq, [r["tau_ext_Pa"] for r in s])(np.log10(qt))), 1)
        out[f"tau_e_at_{qt}"] = te
    # monotone bound for 330 um at 1 uL/s: Det evaluated at the 440 Pa floor instead of its own tau_e
    s = sorted([r for r in rows if r["needle_um"] == 330.0], key=lambda r: r["Q_uLs"])
    lq = np.log10([r["Q_uLs"] for r in s])
    dst = float(PchipInterpolator(lq, [r["Dst_pct"] for r in s])(0.0)) / 100
    det_floor = float(C.det(440.0, 1.43e-3, 0.55))
    out["bound_330_at_1uLs_Dt_pct"] = round(100 * (1 - (1 - D0) * (1 - det_floor) * (1 - dst)), 3)
    # 2. tau_e along the recalibrated Dt = 20 % boundary
    gp_dt, gp_q, z = S.fit_all(rows, check=True)
    gp_te, zt = S.fit_field("tau_ext_Pa", rows, log10=True)
    P = np.linspace(*S.P_TRAINED, 3751)
    bt = {}
    for d in (200.0, 250.0, 290.0, 330.0):
        dt = S.total_from_dn(gp_dt.predict(z(P, np.full_like(P, d))))
        i = int(np.flatnonzero(dt >= S.DT_MAX)[0])
        bt[d] = dict(P_kPa=round(float(P[i]), 1), tau_e_Pa=round(float(10 ** gp_te.predict(zt([P[i]], [d]))[0]), 0))
    out["boundary_tau_e"] = bt
    # 3. kernel / alpha variants: boundaries and membership of the grid cases
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows]); mu, sd = X.mean(0), X.std(0)
    dn = np.array([r["Dn_pct"] for r in rows])
    variants = {"Matern 5/2, 1e-6": (2.5, 1e-6), "Matern 5/2, 1e-4": (2.5, 1e-4), "Matern 5/2, 1e-2": (2.5, 1e-2),
                "Matern 3/2, 1e-6": (1.5, 1e-6), "Matern 3/2, 1e-4": (1.5, 1e-4), "Matern 3/2, 1e-2": (1.5, 1e-2)}
    sens = {}
    for name, (nu, a) in variants.items():
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=nu)
        g = S._fit((X - mu) / sd, dn, kernel=k, alpha=a)
        b = {}
        for d in (200.0, 250.0, 290.0, 330.0):
            Xz = (np.column_stack([P, np.full_like(P, d)]) - mu) / sd
            dt = S.total_from_dn(g.predict(Xz))
            b[d] = round(float(P[int(np.flatnonzero(dt >= S.DT_MAX)[0])]), 1)
        sens[name] = b
    out["boundary_by_surrogate_setting"] = sens
    ref = sens["Matern 5/2, 1e-6"]
    out["boundary_max_shift_kPa"] = round(max(abs(v[d] - ref[d]) for v in sens.values() for d in ref), 1)
    # grid cases nearest to the recalibrated boundary and their distance from it
    out["grid_margin_to_limit_pp"] = sorted(
        [(r["needle_um"], r["pressure_kPa"], round(r["Dt_pct"] - S.DT_MAX, 2)) for r in rows if r["Q_uLs"] >= S.Q_MIN],
        key=lambda t: abs(t[2]))[:4]
    # 4. measured-flow operating pressure of the 200 um nozzle (80.7 kPa, window_analyses.json)
    p_meas = json.loads((DATA / "analyses" / "window_analyses.json").read_text())["flow"]["op_pressure_200um_1uLs_kPa"]["measured"]
    out["Dt_at_measured_op_pressure_200um"] = round(float(S.total_from_dn(gp_dt.predict(z([p_meas], [200.0])))[0]), 2)
    out["Dt_at_model_op_pressure_200um"] = round(float(S.total_from_dn(gp_dt.predict(z([72.7], [200.0])))[0]), 2)
    # 5. friction sensitivity: the Dt = 20 % boundary when the developed nozzle friction is scaled by
    #    1 +/- 1 %, the entrance residual following it (Eq. 17); flow along each diameter by PCHIP in log10 P.
    from l8_damage import compute_Dst, K_VIS, N_EXP, LTIP
    fr = {}
    for d in (200.0, 250.0, 330.0):
        s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["pressure_kPa"])
        lp = np.log10([r["pressure_kPa"] for r in s_]); lq = np.log10([r["Q_uLs"] for r in s_])
        qf = PchipInterpolator(lp, lq)
        R = d * 0.5e-6
        res = {}
        for eps in (-0.01, 0.0, 0.01):
            PP = np.linspace(150.0, 400.0, 1001)
            dts = []
            for P_ in PP:
                Q = 10 ** float(qf(np.log10(P_))) * 1e-9
                gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * Q / (np.pi * R ** 3)
                dptip = (1 + eps) * 2 * LTIP * K_VIS * gdot ** N_EXP / R
                pen = max(P_ * 1e3 - dptip, 0.0)
                det_ = float(C.det(3 * (N_EXP + 1) / 8 * pen, 1.43e-3, 0.55))
                dst_ = compute_Dst(dptip, R)
                dts.append(100 * (1 - (1 - D0) * (1 - det_) * (1 - dst_)))
            dts = np.array(dts)
            k = np.flatnonzero(dts >= 20.0)
            res[eps] = None if k.size == 0 else float(PP[k[0]])
        fr[d] = dict(base=round(res[0.0], 1), minus1=round(res[-0.01], 1), plus1=round(res[0.01], 1))
    out["friction_1pct_boundary_kPa"] = fr
    # 5b. the same friction perturbation at matched flow (1 and 17.94 uL/s): the operating pressure of each nozzle
    #     comes from the solved cases (PCHIP of log10 P in log10 Q) and does not move; the 1 % change acts on the
    #     friction subtracted in Eq. 17 and, through it, on the entrance stress and the shear damage.
    fm = {}
    for qt in (1.0, 17.94):
        cell = {}
        for d in DIAMS:
            s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["Q_uLs"])
            lq = np.log10([r["Q_uLs"] for r in s_]); lp = np.log10([r["pressure_kPa"] for r in s_])
            if not (lq[0] <= np.log10(qt) <= lq[-1]):
                continue
            P_ = 10 ** float(PchipInterpolator(lq, lp)(np.log10(qt)))
            R = d * 0.5e-6
            gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * qt * 1e-9 / (np.pi * R ** 3)
            v = {"P_kPa": round(P_, 2)}
            for eps, key in ((-0.01, "minus1"), (0.0, "base"), (0.01, "plus1")):
                dptip = (1 + eps) * 2 * LTIP * K_VIS * gdot ** N_EXP / R
                pen = max(P_ * 1e3 - dptip, 0.0)
                te = 3 * (N_EXP + 1) / 8 * pen
                det_ = float(C.det(te, 1.43e-3, 0.55)); dst_ = compute_Dst(dptip, R)
                v[key] = round(100 * (1 - (1 - D0) * (1 - det_) * (1 - dst_)), 2)
                v[key + "_tau_e_Pa"] = round(te, 0)
            cell[d] = v
        ds = sorted(cell)
        cell["ordering_holds"] = {k: bool(all(cell[a][k] > cell[b][k] for a, b in zip(ds, ds[1:])))
                                  for k in ("minus1", "base", "plus1")}
        fm[qt] = cell
    out["friction_1pct_matched_flow_Dt_pct"] = fm
    # 5c. the same perturbation at the fixed pressure of 200 kPa (Ning's diameter series): flow of each nozzle at
    #     200 kPa from the solved cases (PCHIP of log10 Q in log10 P), held fixed.
    fp = {}
    for d in DIAMS:
        s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["pressure_kPa"])
        lp = np.log10([r["pressure_kPa"] for r in s_]); lq = np.log10([r["Q_uLs"] for r in s_])
        Q = 10 ** float(PchipInterpolator(lp, lq)(np.log10(200.0))) * 1e-9
        R = d * 0.5e-6
        gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * Q / (np.pi * R ** 3)
        v = {"Q_uLs": round(Q * 1e9, 3)}
        for eps, key in ((-0.01, "minus1"), (0.0, "base"), (0.01, "plus1")):
            dptip = (1 + eps) * 2 * LTIP * K_VIS * gdot ** N_EXP / R
            pen = max(200e3 - dptip, 0.0)
            te = 3 * (N_EXP + 1) / 8 * pen
            det_ = float(C.det(te, 1.43e-3, 0.55)); dst_ = compute_Dst(dptip, R)
            v[key] = round(100 * (1 - (1 - D0) * (1 - det_) * (1 - dst_)), 2)
            v[key + "_tau_e_Pa"] = round(te, 0)
        fp[d] = v
    fp["ordering_holds"] = {k: bool(all(fp[a][k] < fp[b][k] for a, b in zip(DIAMS, DIAMS[1:])))
                            for k in ("minus1", "base", "plus1")}
    out["friction_1pct_fixed_200kPa_Dt_pct"] = fp
    # 5d. the same perturbation at every solved case of the grid (solved flow, applied pressure), with the
    #     entrance pressure as a share of the friction it is the residual of, and the dominant term
    cases, per = [], {}
    for r in sorted(rows, key=lambda r: (r["needle_um"], r["pressure_kPa"])):
        d, P_ = r["needle_um"], r["pressure_kPa"]
        R = d * 0.5e-6
        gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * r["Q_uLs"] * 1e-9 / (np.pi * R ** 3)
        fr0 = 2 * LTIP * K_VIS * gdot ** N_EXP / R
        v = {"D_um": d, "P_kPa": P_, "pen_over_friction_pct": round(100 * (P_ * 1e3 - fr0) / fr0, 2)}
        for eps, key in ((-0.01, "minus1"), (0.0, "base"), (0.01, "plus1")):
            dptip = (1 + eps) * fr0
            te = 3 * (N_EXP + 1) / 8 * max(P_ * 1e3 - dptip, 0.0)
            det_ = float(C.det(te, 1.43e-3, 0.55)); dst_ = compute_Dst(dptip, R)
            v[key] = round(100 * (1 - (1 - D0) * (1 - det_) * (1 - dst_)), 2)
            v[key + "_dominant"] = "entrance" if det_ > dst_ else "shear"
        assert abs(v["base"] - r["Dt_pct"]) < 0.01, (d, P_, v["base"], r["Dt_pct"])
        cases.append(v)
    for d in DIAMS:
        cd = [c for c in cases if c["D_um"] == d]
        per[d] = {"pen_over_friction_pct_min": min(c["pen_over_friction_pct"] for c in cd),
                  "pen_over_friction_pct_max": max(c["pen_over_friction_pct"] for c in cd),
                  "shear_dominated_kPa": {k: [c["P_kPa"] for c in cd if c[k + "_dominant"] == "shear"]
                                          for k in ("minus1", "base", "plus1")},
                  "plus1_Dt_monotonic_decreasing": bool(all(a["plus1"] > b["plus1"] for a, b in zip(cd, cd[1:])))}
    out["friction_1pct_grid"] = {"per_nozzle": per, "cases": cases}
    # 5e. the friction change at which each ordering is first lost (steps of 0.01 %, up to +-10 %), same points as 5b/5c
    def dt_eps(d, P_kPa, Q_uLs, eps):
        R = d * 0.5e-6
        gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * Q_uLs * 1e-9 / (np.pi * R ** 3)
        dptip = (1 + eps) * 2 * LTIP * K_VIS * gdot ** N_EXP / R
        te = 3 * (N_EXP + 1) / 8 * max(P_kPa * 1e3 - dptip, 0.0)
        return 100 * (1 - (1 - D0) * (1 - float(C.det(te, 1.43e-3, 0.55))) * (1 - compute_Dst(dptip, R)))

    def first_loss(points, decreasing):
        res = {}
        for sign, key in ((1, "plus_pct"), (-1, "minus_pct")):
            res[key] = None
            for k in range(1, 1001):
                eps = sign * k * 1e-4
                v = [dt_eps(d, P_, Q_, eps) for d, P_, Q_ in points]
                bad = [i for i, (a, b) in enumerate(zip(v, v[1:])) if not ((a > b) if decreasing else (a < b))]
                if bad:
                    res[key] = round(sign * k * 0.01, 2)
                    res[key.replace("_pct", "_pair_um")] = [points[i][0] for i in (bad[0], bad[0] + 1)]  # the pair that swaps
                    break
        return res

    pts = {}
    for qt in (1.0, 17.94):
        lst = []
        for d in DIAMS:
            s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["Q_uLs"])
            lq = np.log10([r["Q_uLs"] for r in s_]); lp = np.log10([r["pressure_kPa"] for r in s_])
            if lq[0] <= np.log10(qt) <= lq[-1]:
                lst.append((d, 10 ** float(PchipInterpolator(lq, lp)(np.log10(qt))), qt))
        pts[qt] = lst
    fixed = []
    for d in DIAMS:
        s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["pressure_kPa"])
        lp = np.log10([r["pressure_kPa"] for r in s_]); lq = np.log10([r["Q_uLs"] for r in s_])
        fixed.append((d, 200.0, 10 ** float(PchipInterpolator(lp, lq)(np.log10(200.0)))))
    for qt in (1.0, 17.94):                      # the 0 % point must reproduce 5b
        for d, P_, Q_ in pts[qt]:
            assert abs(dt_eps(d, P_, Q_, 0.0) - fm[qt][d]["base"]) < 0.01, (qt, d)
    out["friction_ordering_lost_at"] = {
        "matched_1uLs_all_four": first_loss(pts[1.0], True),
        "matched_1uLs_200_250_330": first_loss([p for p in pts[1.0] if p[0] != 100.0], True),
        "matched_17_94uLs_200_250_330": first_loss(pts[17.94], True),
        "fixed_200kPa_all_four": first_loss(fixed, False)}
    out["friction_ordering_scan_max_pct"] = round(1000 * 1e-4 * 100, 2)   # the range scanned by first_loss, each direction
    (DATA / "analyses" / "extra_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
