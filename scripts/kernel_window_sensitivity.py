#!/usr/bin/env python3
"""The full operating window under the six kernel and regularization settings of the surrogate (Additional file 1, Section S11).

extra_numbers.py moves the Dt = 20 % boundary under these settings (Additional file 1, Table S4). This script adds
the flow boundary (Q = 1 µL/s), the solved and unsolved diameters together (100, 150, 200, 250, 290, 330 µm),
the operating point of each diameter and the window membership of the fifteen FEM checks, each setting against
the reference setting (Matern 5/2, alpha 1e-6, the surrogate of the article) and against the FEM, and the
membership of the 32 solved cases as the surrogate of each setting returns it.

Both outputs are refitted under each setting: Dn and log10 Q, inputs standardized on the 32 cases as in
l8_surrogate.fit_all. Window membership uses the total damage (Eq. 25) below 20 % and the flow rate at or above
1 µL/s, the rule of surrogate_benchmark.py. Writes results/analyses/kernel_window_sensitivity.json.

    python scripts/kernel_window_sensitivity.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import l8_surrogate as S
from l8_paths import DATA
from surrogate_benchmark import checks

VARIANTS = {"Matern 5/2, 1e-6": (2.5, 1e-6), "Matern 5/2, 1e-4": (2.5, 1e-4), "Matern 5/2, 1e-2": (2.5, 1e-2),
            "Matern 3/2, 1e-6": (1.5, 1e-6), "Matern 3/2, 1e-4": (1.5, 1e-4), "Matern 3/2, 1e-2": (1.5, 1e-2)}
REF = "Matern 5/2, 1e-6"
DIAMS = (100.0, 150.0, 200.0, 250.0, 290.0, 330.0)
SOLVED = (100.0, 200.0, 250.0, 330.0)
P = np.linspace(*S.P_TRAINED, 3751)


def first(mask):
    i = np.flatnonzero(mask)
    return None if i.size == 0 else round(float(P[i[0]]), 1)


def main():
    rows = S.load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows]); mu, sd = X.mean(0), X.std(0)
    dn = np.array([r.get("Dn_pct", r["Dt_pct"]) for r in rows])
    lq = np.log10([r["Q_uLs"] for r in rows])
    ck = checks()
    assert len(ck) == 15, f"expected the fifteen FEM checks, found {len(ck)}"
    acc = lambda dn_, q: bool((S.total_from_dn(dn_) < S.DT_MAX) and (q >= S.Q_MIN))
    fem_in = [acc(c["Dt"], c["Q"]) for c in ck]
    grid_in = [acc(r.get("Dn_pct", r["Dt_pct"]), r["Q_uLs"]) for r in rows]   # the 32 solved cases, FEM values
    out = dict(variants={}, n_checks=len(ck), fem_in_window=fem_in,
               # distance of each FEM check from the two limits: Dt total minus 20 pp, and Q above 1 uL/s in %
               fem_checks=[dict(D=c["D"], P=c["P"], Dt_total_minus_limit_pp=round(float(S.total_from_dn(c["Dt"]) - S.DT_MAX), 3),
                                Q_above_limit_pct=round(float(100 * (c["Q"] / S.Q_MIN - 1)), 2)) for c in ck])
    for name, (nu, a) in VARIANTS.items():
        k = lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=nu)
        g_dn = S._fit((X - mu) / sd, dn, kernel=k(), alpha=a)
        g_lq = S._fit((X - mu) / sd, lq, kernel=k(), alpha=a)
        z = lambda p, d: (np.column_stack([np.ravel(p), np.ravel(d)]) - mu) / sd
        per_d = {}
        for d in DIAMS:
            dt = S.total_from_dn(g_dn.predict(z(P, np.full_like(P, d))))
            q = 10 ** g_lq.predict(z(P, np.full_like(P, d)))
            ok = (dt < S.DT_MAX) & (q >= S.Q_MIN)
            op = None if not ok.any() else round(float(P[np.flatnonzero(ok)[np.argmin(dt[ok])]]), 1)
            per_d[str(d)] = dict(damage_boundary_kPa=first(dt >= S.DT_MAX), flow_boundary_kPa=first(q >= S.Q_MIN),
                                 op_kPa=op, window_width_kPa=None if not ok.any() else round(float(P[ok].max() - P[ok].min()), 1))
        cin = [acc(g_dn.predict(z([c["P"]], [c["D"]]))[0], 10 ** g_lq.predict(z([c["P"]], [c["D"]]))[0]) for c in ck]
        gin = [acc(g_dn.predict(z([r["pressure_kPa"]], [r["needle_um"]]))[0],
                   10 ** g_lq.predict(z([r["pressure_kPa"]], [r["needle_um"]]))[0]) for r in rows]
        ops = {d: per_d[str(d)]["op_kPa"] for d in SOLVED if per_d[str(d)]["op_kPa"] is not None}
        dt_op = {d: float(S.total_from_dn(g_dn.predict(z([p], [d])))[0]) for d, p in ops.items()}
        out["variants"][name] = dict(per_diameter=per_d, checks_in_window=cin, grid_in_window=gin,
                                     grid_differing_from_fem=[(r["needle_um"], r["pressure_kPa"], round(float(r["Q_uLs"]), 3)) for r, x, y in
                                                              zip(rows, gin, grid_in) if x != y],
                                     checks_differing_from_fem=int(sum(x != y for x, y in zip(cin, fem_in))),
                                     selected_nozzle=min(dt_op, key=dt_op.get) if dt_op else None)
    ref = out["variants"][REF]
    shift = {}
    for key in ("damage_boundary_kPa", "flow_boundary_kPa", "op_kPa"):
        for grp, ds in (("solved", SOLVED), ("unsolved", (150.0, 290.0))):
            v = [abs(out["variants"][n]["per_diameter"][str(d)][key] - ref["per_diameter"][str(d)][key])
                 for n in VARIANTS for d in ds
                 if out["variants"][n]["per_diameter"][str(d)][key] is not None and ref["per_diameter"][str(d)][key] is not None]
            shift[f"{key}_max_shift_{grp}"] = round(max(v), 1) if v else None
    out["max_shift_against_reference"] = shift
    out["checks_differing_from_reference"] = {n: int(sum(x != y for x, y in zip(v["checks_in_window"], ref["checks_in_window"])))
                                              for n, v in out["variants"].items()}
    out["selected_nozzle_by_setting"] = {n: v["selected_nozzle"] for n, v in out["variants"].items()}
    (DATA / "analyses").mkdir(parents=True, exist_ok=True)
    (DATA / "analyses" / "kernel_window_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(shift, indent=1))
    print("checks differing from the reference setting:", out["checks_differing_from_reference"])
    print("checks differing from the FEM:", {n: v["checks_differing_from_fem"] for n, v in out["variants"].items()})
    print("selected nozzle:", out["selected_nozzle_by_setting"])
    print("grid cases classed unlike the FEM:", {n: v["grid_differing_from_fem"] for n, v in out["variants"].items()})
    for n, v in out["variants"].items():
        print(n, {d: (x["damage_boundary_kPa"], x["flow_boundary_kPa"], x["op_kPa"]) for d, x in v["per_diameter"].items()})


if __name__ == "__main__":
    main()
