"""Per FEM check, the margin to each limit and which method decides differently (Additional file 1, Section S6).

Rebuilds the four emulators of surrogate_benchmark.py on the same 32 cases and, for each of the fifteen FEM
checks, records the margin of the FEM solution to the damage limit (Dt = 20 %) and to the flow limit (1 µL/s),
and which emulator places the check on the other side of the window from the FEM. It reproduces the counts of
differing decisions in surrogate_benchmark.json (asserted) and writes
results/analyses/surrogate_benchmark_per_check.json.

    python scripts/surrogate_benchmark_per_check.py
"""
from __future__ import annotations

import json

import numpy as np

import l8_surrogate as S
import surrogate_benchmark as B
from l8_paths import DATA


def main():
    rows, D, P, dt, lq = B.load()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    y_dt = np.array([r["Dt_pct"] for r in rows]); y_lq = np.log10([r["Q_uLs"] for r in rows])
    full = {"GP": (B.GP(X, y_dt), B.GP(X, y_lq)), "RBF thin-plate": (B.RBF(X, y_dt), B.RBF(X, y_lq)),
            "PCHIP": (B.Tensor(D, P, dt, "pchip"), B.Tensor(D, P, lq, "pchip")),
            "spline": (B.Tensor(D, P, dt, "spline"), B.Tensor(D, P, lq, "spline"))}
    acc = lambda dn, q: bool((S.total_from_dn(dn) < S.DT_MAX) and (q >= S.Q_MIN))
    out = []
    for c in B.checks():
        tot = float(S.total_from_dn(c["Dt"]))
        row = dict(D_um=c["D"], P_kPa=c["P"], set=c["set"], Dt_total_pct=tot, margin_to_damage_limit_pp=tot - S.DT_MAX,
                   Q_uLs=c["Q"], margin_to_flow_limit_pct=100 * (c["Q"] / S.Q_MIN - 1), fem_inside=acc(c["Dt"], c["Q"]),
                   differs=[], error={})
        for k, (fd, fq) in full.items():
            dn_k, q_k = fd(c["D"], c["P"])[0], 10 ** fq(c["D"], c["P"])[0]
            row["error"][k] = dict(Dt_pp=float(S.total_from_dn(dn_k)) - tot, Q_pct=100 * (q_k / c["Q"] - 1))
            if acc(dn_k, q_k) != row["fem_inside"]:
                row["differs"].append(k)
        out.append(row)
    bench = json.loads((DATA / "analyses" / "surrogate_benchmark.json").read_text())
    for k in full:
        n = sum(k in r["differs"] for r in out)
        assert n == bench["checks"][k]["window_disagree"], (k, n, bench["checks"][k]["window_disagree"])
    (DATA / "analyses" / "surrogate_benchmark_per_check.json").write_text(json.dumps(out, indent=1, default=float))
    for r in out:
        print(f"{r['D_um']:5.0f} um {r['P_kPa']:6.1f} kPa {r['set']:6s} Dt {r['Dt_total_pct']:6.2f}% "
              f"(limit {r['margin_to_damage_limit_pp']:+.2f} pp) Q {r['Q_uLs']:8.3f} uL/s "
              f"(flow limit {r['margin_to_flow_limit_pct']:+.2f}%) differs: {', '.join(r['differs']) or '-'}")


if __name__ == "__main__":
    main()
