#!/usr/bin/env python3
"""The Gaussian process against interpolants of the same 32 solved cases (Additional file 1, Section S6).

Every method sees the same training data and is scored on the same tests:

  held-out splits   one case, one whole diameter, one whole pressure left out (the tests of
                    surrogate_validation.py); a tensor-product interpolant needs a complete grid, so single
                    cases are left out of a thin-plate radial basis function instead, the scattered-data
                    counterpart (Tables S5 and S6);
  FEM checks        the eleven conditions of prespecify_fem_checks.py and the four of
                    prespecify_boundary_checks.py, none used in any fit;
  decisions         the Dt = 20 % boundary and the operating point of every diameter, and the nozzle
                    with the lowest damage at the operating points (Table S7);
  cost              time per query.

Interpolants: tensor PCHIP in (diameter, log10 P), bicubic spline in (diameter, log10 P), thin-plate RBF in
standardized (P, D). Damage targets are Dn, the damage of the published law on the grid; the total damage Dt
is the same affine map of Dn for every method, so the comparison carries over. Writes
results/analyses/surrogate_benchmark.json.

    python scripts/surrogate_benchmark.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import time

import numpy as np
from scipy.interpolate import PchipInterpolator, RBFInterpolator, RectBivariateSpline

import l8_surrogate as S
from l8_paths import DATA

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "fem" / "runs"
P_SWEEP = np.linspace(*S.P_TRAINED, 3751)


def load():
    rows = [dict(r, Dt_pct=r.get("Dn_pct", r["Dt_pct"])) for r in S.load_grid()]   # Dn, which the surrogate emulates
    D = np.array(sorted({r["needle_um"] for r in rows}))
    P = np.array(sorted({r["pressure_kPa"] for r in rows}))
    get = {(r["needle_um"], r["pressure_kPa"]): r for r in rows}
    dt = np.array([[get[(d, p)]["Dt_pct"] for p in P] for d in D])
    lq = np.array([[np.log10(get[(d, p)]["Q_uLs"]) for p in P] for d in D])
    return rows, D, P, dt, lq


class Tensor:
    """Tensor-product interpolant on a (D, log P) grid; PCHIP or bicubic spline."""

    def __init__(self, D, P, Z, kind):
        self.D, self.lP, self.Z, self.kind = D, np.log10(P), Z, kind
        if kind == "spline":
            kx, ky = min(3, len(D) - 1), min(3, len(P) - 1)
            self.s = RectBivariateSpline(D, self.lP, Z, kx=kx, ky=ky)

    def __call__(self, d, p):
        d, p = np.atleast_1d(d).astype(float), np.atleast_1d(p).astype(float)
        if self.kind == "spline":
            return self.s.ev(d, np.log10(p))
        cols = np.array([PchipInterpolator(self.lP, self.Z[i], extrapolate=True)(np.log10(p)) for i in range(len(self.D))])
        return np.array([PchipInterpolator(self.D, cols[:, k], extrapolate=True)(d[k] if d.size > 1 else d[0])
                         for k in range(len(p))])


class RBF:
    def __init__(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.f = RBFInterpolator((X - self.mu) / self.sd, y, kernel="thin_plate_spline")

    def __call__(self, d, p):
        X = np.column_stack([np.ravel(p), np.ravel(d)]).astype(float)
        return self.f((X - self.mu) / self.sd)


class GP:
    def __init__(self, X, y):
        self.m = S.FoldModel(X, y)

    def __call__(self, d, p):
        return self.m.predict(np.column_stack([np.ravel(p), np.ravel(d)]).astype(float))


def holdouts(rows, D, P, dt, lq):
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    y_dt = np.array([r["Dt_pct"] for r in rows]); y_lq = np.log10([r["Q_uLs"] for r in rows])
    out = {}
    # single cases: GP and RBF (scattered)
    for name, cls in (("GP", GP), ("RBF thin-plate", RBF)):
        e_dt, e_q = [], []
        for i in range(len(X)):
            m = np.arange(len(X)) != i
            e_dt.append(cls(X[m], y_dt[m])(X[i, 1], X[i, 0])[0] - y_dt[i])
            e_q.append(10 ** (cls(X[m], y_lq[m])(X[i, 1], X[i, 0])[0] - y_lq[i]) - 1)
        out.setdefault(name, {})["case"] = dict(Dt_MAE=np.mean(np.abs(e_dt)), Dt_max=np.max(np.abs(e_dt)),
                                                 Q_MAPE=100 * np.mean(np.abs(e_q)), Q_max=100 * np.max(np.abs(e_q)))
    # whole diameter / whole pressure: every method
    for split in ("diameter", "pressure"):
        levels = D if split == "diameter" else P
        errs = {k: dict(dt=[], q=[]) for k in ("GP", "RBF thin-plate", "PCHIP", "spline")}
        for lev in levels:
            keep = (X[:, 1] != lev) if split == "diameter" else (X[:, 0] != lev)
            test = ~keep
            Dk = D[D != lev] if split == "diameter" else D
            Pk = P[P != lev] if split == "pressure" else P
            dtk = dt[D != lev] if split == "diameter" else dt[:, P != lev]
            lqk = lq[D != lev] if split == "diameter" else lq[:, P != lev]
            models = {"GP": (GP(X[keep], y_dt[keep]), GP(X[keep], y_lq[keep])),
                      "RBF thin-plate": (RBF(X[keep], y_dt[keep]), RBF(X[keep], y_lq[keep])),
                      "PCHIP": (Tensor(Dk, Pk, dtk, "pchip"), Tensor(Dk, Pk, lqk, "pchip")),
                      "spline": (Tensor(Dk, Pk, dtk, "spline"), Tensor(Dk, Pk, lqk, "spline"))}
            for k, (fd, fq) in models.items():
                pd_ = fd(X[test, 1], X[test, 0]); pq = fq(X[test, 1], X[test, 0])
                errs[k]["dt"] += list(pd_ - y_dt[test]); errs[k]["q"] += list(10 ** (pq - y_lq[test]) - 1)
        for k, e in errs.items():
            out.setdefault(k, {})[split] = dict(Dt_MAE=float(np.mean(np.abs(e["dt"]))), Dt_max=float(np.max(np.abs(e["dt"]))),
                                                Q_MAPE=float(100 * np.mean(np.abs(e["q"]))), Q_max=float(100 * np.max(np.abs(e["q"]))))
            # interior levels test interpolation across a missing level, edge levels extrapolation
            nlev = len(levels); per = len(e["dt"]) // nlev
            for part, idx in (("interior", range(1, nlev - 1)), ("edge", (0, nlev - 1))):
                sel = [j for li in idx for j in range(li * per, (li + 1) * per)]
                out[k][f"{split}_{part}"] = dict(Dt_MAE=float(np.mean(np.abs(np.array(e["dt"])[sel]))),
                                                 Dt_max=float(np.max(np.abs(np.array(e["dt"])[sel]))),
                                                 Q_MAPE=float(100 * np.mean(np.abs(np.array(e["q"])[sel]))))
    return out


def checks():
    rows = []
    # Dt_fem_pct of both scored files is Dn from the canonical chain (l8_damage.derive) at the solved P and Q, the
    # quantity the grid and the surrogate carry; the solver's own Dt_pct column is a different estimate and is not used
    for run, s in (("2026-09-23_independent-checks", "eleven"), ("2026-10-31_boundary-checks", "four")):
        f = RUNS / run / "checks_vs_surrogate.csv"
        if f.exists():
            rows += [dict(D=float(r["needle_um"]), P=float(r["pressure_kPa"]), Dt=float(r["Dt_fem_pct"]),
                          Q=float(r["Q_fem_uLs"]), set=s) for r in csv.DictReader(open(f))]
    return rows


DN_MAX = (S.DT_MAX - 100 * S.D0) / (1 - S.D0)    # Dt = 20 % on the recalibrated law, as a contour of Dn


def decisions(f_dt, f_lq, dt_max=DN_MAX, q_min=S.Q_MIN):
    out = {}
    for d in (100.0, 150.0, 200.0, 250.0, 290.0, 330.0):
        dtv = f_dt(np.full_like(P_SWEEP, d), P_SWEEP); q = 10 ** f_lq(np.full_like(P_SWEEP, d), P_SWEEP)
        ok = (dtv < dt_max) & (q >= q_min)
        op = None if not ok.any() else (round(float(P_SWEEP[np.flatnonzero(ok)[np.argmin(dtv[ok])]]), 1),
                                        round(float(dtv[ok].min()), 3))
        hit = np.flatnonzero(dtv >= dt_max)
        out[d] = dict(op=op, boundary=None if hit.size == 0 else round(float(P_SWEEP[hit[0]]), 1))
    return out


def main():
    rows, D, P, dt, lq = load()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    y_dt = np.array([r["Dt_pct"] for r in rows]); y_lq = np.log10([r["Q_uLs"] for r in rows])
    full = {"GP": (GP(X, y_dt), GP(X, y_lq)), "RBF thin-plate": (RBF(X, y_dt), RBF(X, y_lq)),
            "PCHIP": (Tensor(D, P, dt, "pchip"), Tensor(D, P, lq, "pchip")),
            "spline": (Tensor(D, P, dt, "spline"), Tensor(D, P, lq, "spline"))}
    ck = checks()
    res = dict(holdouts=holdouts(rows, D, P, dt, lq), checks={}, decisions={}, query_ms={})
    for k, (fd, fq) in full.items():
        e_dt = [fd(c["D"], c["P"])[0] - c["Dt"] for c in ck]
        e_q = [10 ** (fq(c["D"], c["P"])[0]) / c["Q"] - 1 for c in ck]
        unsolved = [i for i, c in enumerate(ck) if c["D"] in (150.0, 290.0)]
        res["checks"][k] = dict(n=len(ck), Dt_MAE=float(np.mean(np.abs(e_dt))), Dt_max=float(np.max(np.abs(e_dt))),
                                Q_MAPE=float(100 * np.mean(np.abs(e_q))), Q_max=float(100 * np.max(np.abs(e_q))),
                                Dt_MAE_unsolved_D=float(np.mean(np.abs(np.array(e_dt)[unsolved]))))
        res["decisions"][k] = decisions(fd, fq)
        # How each method classifies the checks and which nozzle it selects. Window membership on the total
        # (recalibrated law) and the flow limit, against the FEM's; the selected nozzle has the lowest damage at its
        # operating point among the four solved diameters.
        acc = lambda dn, q: (S.total_from_dn(dn) < S.DT_MAX) and (q >= S.Q_MIN)
        res["checks"][k]["window_disagree"] = int(sum(
            acc(fd(c["D"], c["P"])[0], 10 ** fq(c["D"], c["P"])[0]) != acc(c["Dt"], c["Q"]) for c in ck))
        ops = {d: v["op"] for d, v in res["decisions"][k].items() if d in (100.0, 200.0, 250.0, 330.0) and v["op"]}
        res["decisions_selected_nozzle"] = res.get("decisions_selected_nozzle", {})
        res["decisions_selected_nozzle"][k] = min(ops, key=lambda d: ops[d][1]) if ops else None
        t0 = time.perf_counter()
        for _ in range(200):
            fd(np.array([215.0]), np.array([123.0]))
        res["query_ms"][k] = 1e3 * (time.perf_counter() - t0) / 200
    (DATA / "analyses").mkdir(parents=True, exist_ok=True)
    (DATA / "analyses" / "surrogate_benchmark.json").write_text(json.dumps(res, indent=1, default=float))
    for k in full:
        h = res["holdouts"][k]
        print(f"{k:15s} checks n={res['checks'][k]['n']} Dt MAE {res['checks'][k]['Dt_MAE']:.3f} max {res['checks'][k]['Dt_max']:.3f}"
              f" (unsolved D {res['checks'][k]['Dt_MAE_unsolved_D']:.3f}) | Q MAPE {res['checks'][k]['Q_MAPE']:.2f} max {res['checks'][k]['Q_max']:.2f}"
              f" | {res['query_ms'][k]:.3f} ms/query")
        for s, v in h.items():
            print(f"     holdout {s:8s} Dt MAE {v['Dt_MAE']:.3f} max {v['Dt_max']:.3f} | Q MAPE {v['Q_MAPE']:.1f} max {v.get('Q_max', float('nan')):.1f}")
        print("     decisions", {d: (v['op'], v['boundary']) for d, v in res['decisions'][k].items()})
        print(f"     window decisions differing from the FEM at the checks: {res['checks'][k]['window_disagree']} of "
              f"{res['checks'][k]['n']}; selected nozzle {res['decisions_selected_nozzle'][k]}")


if __name__ == "__main__":
    main()
