#!/usr/bin/env python3
"""The shear-damage estimate of Ning et al. (2018), Fig. 6, and the offset between model and measured damage.

Their Eq. 9 (Eq. 21 here), evaluated exactly with their inputs, gives far less shear damage than
the estimate they plot. Evaluated by the trapezoid rule on a radial grid, the wall node, where the
velocity is zero and the exposure time unbounded, counts as fully damaged and adds a term that
does not depend on pressure; the script finds the number of intervals that reproduces their
values for both cell lines, and splits the gap between the model and their measured damage into
this offset and the residuals of their fit. Their Fig. 6 values were read from the published
figure. Writes results/validation/ning_fig6_shear_offset.json.

    python scripts/validation/reproduce_ning_fig6_shear.py
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import l8_damage as dmg  # noqa: E402
from l8_paths import CANONICAL_GRID, require  # noqa: E402

L, R = 10e-3, 100e-6
P_KPA = np.array([50.0, 100.0, 200.0, 400.0])
CELLS = {
    # shear law (a1, b1, c1): their Eqs. 16 and 15; K, n: their Section 4.2;
    # P_en: their Table 2 (Pa); fig6: their Fig. 6B (L8) and 6A (RSC96) shear points (%)
    "L8":    dict(law=(3.2e-4, 0.61, 0.52), K=25.60, n=0.35,
                  Pen=[867.0, 1773.0, 3734.0, 18246.0], fig6=[5.82, 5.55, 5.36, 5.16]),
    "RSC96": dict(law=(3.9e-4, 0.48, 0.71), K=20.81, n=0.37,
                  Pen=[804.0, 1714.0, 3373.0, 18334.0], fig6=[5.68, 5.43, 5.21, 5.19]),
}
# L8 damage measured immediately after printing, Ø200 µm (their Section 4.6), and their
# extensional stresses (Table 2), on which Eq. 19 was fitted.
MEASURED_L8 = np.array([6.99, 11.70, 15.53, 23.08])
TAU_E_NING_L8 = np.array([440.0, 899.0, 1893.0, 9251.0])
OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "validation" / "ning_fig6_shear_offset.json"


def ds_point(r, dP, law, K, n):
    a1, b1, c1 = law
    tau = dP * r / (2 * L)
    V = n / (n + 1) * (dP * R / (2 * K * L)) ** (1 / n) * R * (1 - (r / R) ** ((n + 1) / n))
    with np.errstate(divide="ignore"):
        t = np.where(V > 0, L / np.where(V > 0, V, 1.0), np.inf)
    return 1 - np.exp(-a1 * tau ** b1 * t ** c1)


def exact(dP, law, K, n):
    """Their Eq. 9 on panels that halve towards the wall, as in l8_damage.compute_Dst."""
    x, w = np.polynomial.legendre.leggauss(400)
    edges = [0.0] + [R * (1.0 - 0.5 ** k) for k in range(1, 41)] + [R]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        rv = 0.5 * (hi - lo) * (x + 1) + lo
        total += float(np.dot(0.5 * (hi - lo) * w, rv * ds_point(rv, dP, law, K, n)))
    return 2 * total / R ** 2 * 100


def grid(dP, law, K, n, N):
    r = np.linspace(0.0, R, N + 1)
    f = r * ds_point(r, dP, law, K, n)
    return 2 * float(np.sum((f[1:] + f[:-1]) * np.diff(r)) / 2) / R ** 2 * 100


def eq19(tau_e):
    return (1 - np.exp(-dmg.A2 * tau_e ** dmg.B2)) * 100


def main() -> int:
    out = {"source": "Ning et al. 2018, ACS Biomater. Sci. Eng. 4:3906-3918, "
                     "doi:10.1021/acsbiomaterials.8b00714, Eqs. 4-9, 15, 16, 19; Table 2; Fig. 6",
           "pressure_kPa": P_KPA.tolist(), "cells": {}}
    for name, c in CELLS.items():
        dP = P_KPA * 1e3 - np.array(c["Pen"])
        ex = np.array([exact(d, c["law"], c["K"], c["n"]) for d in dP])
        fig = np.array(c["fig6"])
        if name == "L8":   # the manuscript's own implementation must give the same integral
            mine = np.array([dmg.compute_Dst(d, R) * 100 for d in dP])
            assert np.allclose(mine, ex, rtol=1e-6), (mine, ex)
        sweep = {N: np.array([grid(d, c["law"], c["K"], c["n"], N) for d in dP]) for N in range(10, 61)}
        rms = {N: float(np.sqrt(np.mean((v - fig) ** 2))) for N, v in sweep.items()}
        best = min(rms, key=lambda k: rms[k])
        out["cells"][name] = {
            "fig6_shear_pct": fig.tolist(), "exact_eq9_pct": ex.round(4).tolist(),
            "fig6_minus_exact_pp": (fig - ex).round(3).tolist(),
            "exact_fall_50_to_400": round(float(ex[0] / ex[-1]), 2),
            "best_grid_intervals": best, "best_grid_points": best + 1, "best_grid_pct": sweep[best].round(3).tolist(),
            "best_grid_rms_pp": round(rms[best], 3),
            "grid20_pct": sweep[20].round(3).tolist(), "grid20_rms_pp": round(rms[20], 3),
        }
        print(f"{name:6s} Fig. 6 {fig}  exact {ex.round(3)}  offset {(fig - ex).round(2)}  "
              f"best N={best} rms {rms[best]:.2f} pp; N=20 rms {rms[20]:.2f} pp")

    # The model against the measured L8 damage, on the canonical grid.
    with open(require(CANONICAL_GRID, "canonical grid")) as f:
        rows = {float(r["pressure_kPa"]): r for r in csv.DictReader(f)
                if r["needle_um"] == "200" and r["mesh_M"] == "12" and r["inertia"] == "1"}
    col = lambda k: np.array([float(rows[P][k]) for P in P_KPA])
    fig = np.array(CELLS["L8"]["fig6"])
    gap = MEASURED_L8 - col("Dt_pct")
    offset = fig - col("Dst_pct")
    fit_res = (MEASURED_L8 - fig) - eq19(TAU_E_NING_L8)
    out["model_vs_measured_L8"] = {
        "model_Dt_pct": col("Dt_pct").round(3).tolist(), "measured_pct": MEASURED_L8.tolist(),
        "gap_pp": gap.round(3).tolist(), "shear_offset_pp": offset.round(3).tolist(),
        "gap_minus_offset_pp": (gap - offset).round(3).tolist(),
        "eq19_fit_residual_pp": fit_res.round(3).tolist(),
        "model_tau_e_Pa": col("tau_ext_Pa").round(1).tolist(), "ning_tau_e_Pa": TAU_E_NING_L8.tolist(),
    }
    print("gap measured - model      ", gap.round(2))
    print("shear offset (Fig.6B - Eq.21)", offset.round(2))
    print("gap - offset              ", (gap - offset).round(2))
    print("their Eq. 19 fit residuals", fit_res.round(2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
