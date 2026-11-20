#!/usr/bin/env python3
"""Calibration of the damage law against the damage Ning et al. (2018) measured, with this model's shear integral.

Ning et al. fitted the extensional coefficients a2, b2 to the damage they measured through the 200 µm nozzle
at 50-400 kPa, after subtracting their own shear estimate of about 5 %. Evaluated exactly (Eq. 21), the same
shear law gives 0.16-0.95 % at those conditions, so the published coefficients leave the measured damage
under-predicted by 2.1-6.6 percentage points. Three readings of that gap are compared:

  P   published a2, b2, nothing added;
  O   published a2, b2 plus a damage D0 that depends on neither pressure nor diameter,
      Dt = D0 + (1 - D0) Dn (Eq. 25); one fitted parameter;
  R   a2, b2 refitted to the same four points; two fitted parameters.

The damage Ning et al. measured through the 100, 250 and 330 µm nozzles at 200 kPa (their Fig. 7) enters no
fit and is the out-of-sample test. The same chain serves the RSC96 Schwann-cell suspension, whose rheology and
damage law they report alongside L8. Additional file 1, Section S8 and Table S6. Writes
results/analyses/calibration_variants.json.

    python scripts/l8_calibration.py
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import curve_fit

import ning2018_data as ning
from l8_damage import derive as derive_published
from l8_geometry import L_NEEDLE
from l8_paths import CANONICAL_GRID, DATA

OUT = DATA / "analyses"
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)
LTIP = L_NEEDLE


def chain(P_kPa: float, Q_uLs: float, R: float, line: str = "L8", fric_scale: float = 1.0) -> dict:
    """Shear and extensional damage for one case, any cell line in `ning2018_data`.

    Same chain as `l8_damage.derive` (Rabinowitsch wall stress from Q, developed nozzle drop,
    entrance residual, Cogswell stress, area-averaged shear damage on the analytic profile);
    `selftest()` checks the two agree on all 32 grid cases for L8. `fric_scale` multiplies the developed
    nozzle drop subtracted from the applied pressure (the friction sensitivity test; 1.0 = the chain).
    """
    K, n = ning.RHEOLOGY[line]["K"], ning.RHEOLOGY[line]["n"]
    a1, b1, c1 = (ning.LAW[line][k] for k in ("a1", "b1", "c1"))
    Q = Q_uLs * 1e-9
    gdot = (3 * n + 1) / (4 * n) * 4 * Q / (math.pi * R ** 3)
    tw = K * gdot ** n
    dP_tip = fric_scale * 2 * LTIP * tw / R
    P_en = max(P_kPa * 1e3 - dP_tip, 0.0)
    tau_e = 3 * (n + 1) / 8 * P_en
    C = (dP_tip / (2 * K * LTIP)) ** (1 / n)
    alpha = (n + 1) / n
    edges = [0.0] + [R * (1.0 - 0.5 ** k) for k in range(1, 41)] + [R]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        rv = 0.5 * (hi - lo) * (_GL_X + 1.0) + lo
        w = 0.5 * (hi - lo) * _GL_W
        tau_s = dP_tip * rv / (2 * LTIP)
        V = np.maximum((n / (n + 1)) * C * (R ** alpha - rv ** alpha), 1e-14)
        integ = rv * (1.0 - np.exp(-a1 * tau_s ** b1 * (LTIP / V) ** c1))
        total += float(np.dot(w, np.where(tau_s < 1e-12, 0.0, integ)))
    Dst = min(2.0 / R ** 2 * total, 1.0)
    return dict(tau_w_Pa=tw, P_en_kPa=P_en / 1e3, tau_e_Pa=tau_e, Dst=Dst)


def det(tau_e, a2, b2):
    return 1.0 - np.exp(-a2 * np.asarray(tau_e, float) ** b2)


def total(tau_e, Dst, a2, b2, D0=0.0):
    """Cascade of a pressure-independent term D0, the entrance term and the nozzle shear term."""
    return 1.0 - (1.0 - D0) * (1.0 - det(tau_e, a2, b2)) * (1.0 - np.asarray(Dst, float))


def grid_rows(line="L8"):
    import csv
    rows = [r for r in csv.DictReader(open(CANONICAL_GRID))
            if int(float(r["mesh_M"])) == 12 and int(float(r["inertia"])) == 1]
    out = []
    for r in rows:
        D, P, Q = float(r["needle_um"]), float(r["pressure_kPa"]), float(r["Q_uLs"])
        out.append(dict(D=D, P=P, Q=Q, **chain(P, Q, D * 0.5e-6, line)))
    return out


def fit_variants(cal: list[dict], measured_pct: np.ndarray, law: dict) -> dict:
    """P, O and R on the four calibration points. Standard errors from the fit covariance."""
    te = np.array([c["tau_e_Pa"] for c in cal]); ds = np.array([c["Dst"] for c in cal])
    y = measured_pct / 100.0
    a2, b2 = law["a2"], law["b2"]
    v = {"P": dict(a2=a2, b2=b2, D0=0.0, n_par=0)}
    (d0,), cov = curve_fit(lambda _t, d0: total(te, ds, a2, b2, d0), te, y, p0=(0.05,))
    v["O"] = dict(a2=a2, b2=b2, D0=float(d0), n_par=1, se=dict(D0=float(math.sqrt(cov[0, 0]))))
    # R: invert the cascade for the entrance target, then fit a2, b2 to it
    target = 1.0 - (1.0 - y) / (1.0 - ds)
    (ra, rb), cov = curve_fit(lambda t, a, b: det(t, a, b), te, target, p0=(a2, b2), maxfev=20000)
    v["R"] = dict(a2=float(ra), b2=float(rb), D0=0.0, n_par=2,
                  se=dict(a2=float(math.sqrt(cov[0, 0])), b2=float(math.sqrt(cov[1, 1]))))
    for k, p in v.items():
        pred = 100 * total(te, ds, p["a2"], p["b2"], p["D0"])
        res = pred - measured_pct
        p.update(fit_pred_pct=pred.round(3).tolist(), fit_resid_pp=res.round(3).tolist(),
                 fit_rms_pp=float(np.sqrt(np.mean(res ** 2))))
    return v


def selftest() -> None:
    """The generic chain reproduces l8_damage.derive on every grid case (L8)."""
    worst = 0.0
    for r in grid_rows("L8"):
        ref = derive_published(r["P"], r["Q"], r["D"] * 0.5e-6)
        mine = 100 * total(r["tau_e_Pa"], r["Dst"], ning.LAW["L8"]["a2"], ning.LAW["L8"]["b2"])
        worst = max(worst, abs(float(mine) - ref["Dn_pct"]))
    assert worst < 1e-9, worst
    print(f"selftest: generic chain = l8_damage.derive on 32 cases (max |dDt| {worst:.1e} pp)")


def main() -> None:
    selftest()
    rows = grid_rows("L8")
    by = {(r["D"], r["P"]): r for r in rows}
    series = ning.DAMAGE_PRESSURE_SERIES_200UM["L8"]
    cal = [by[(200.0, float(p))] for p in series]
    v = fit_variants(cal, np.array(list(series.values()), float), ning.LAW["L8"])
    oos = ning.DAMAGE_DIAMETER_SERIES_200KPA["L8"]
    test_d = [100, 250, 330]
    for k, p in v.items():
        pred = [100 * float(total(by[(float(d), 200.0)]["tau_e_Pa"], by[(float(d), 200.0)]["Dst"],
                                  p["a2"], p["b2"], p["D0"])) for d in test_d]
        res = [pr - oos[d][0] for pr, d in zip(pred, test_d)]
        p.update(oos_diameters_um=test_d, oos_pred_pct=np.round(pred, 3).tolist(),
                 oos_meas_pct=[oos[d][0] for d in test_d], oos_resid_pp=np.round(res, 3).tolist(),
                 oos_rms_pp=float(np.sqrt(np.mean(np.square(res)))),
                 oos_inside_bar=[bool(oos[d][1] <= pr <= oos[d][2]) for pr, d in zip(pred, test_d)])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_variants.json").write_text(json.dumps(v, indent=2))
    for k, p in v.items():
        se = p.get("se", {})
        print(f"{k}: a2={p['a2']:.3e} b2={p['b2']:.3f} D0={100*p['D0']:.2f}% {se and {a: round(b, 5) for a, b in se.items()}}")
        print(f"   calibration rms {p['fit_rms_pp']:.2f} pp, residuals {p['fit_resid_pp']}")
        print(f"   out-of-sample (200 kPa; {test_d} um): pred {p['oos_pred_pct']} vs meas {p['oos_meas_pct']}"
              f" -> rms {p['oos_rms_pp']:.2f} pp, inside error bar {p['oos_inside_bar']}")


if __name__ == "__main__":
    main()
