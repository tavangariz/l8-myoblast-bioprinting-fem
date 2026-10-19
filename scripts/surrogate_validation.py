#!/usr/bin/env python3
"""Validation of the Gaussian process surrogate on the 32-case grid.

Held-out tests, with the input standardization and hyperparameters refitted inside every fold:
one case, a whole nozzle diameter or a whole applied pressure left out. For each output: mean
absolute, root-mean-square and largest error, the operating-window decisions, and the coverage of
the nominal 95 % interval. Also the same splits for linear, quadratic and cubic polynomial
surfaces, a sensitivity study over kernel smoothness and alpha, a learning curve, and fit and
query timings. Writes surrogate_validation.json, fold_predictions.csv, blocked_holdouts.csv and
sensitivity.csv to results/surrogate/.

    python scripts/surrogate_validation.py
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import platform
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import l8_surrogate as S

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "surrogate"
Z95 = 1.959963985
LEARNING_REPEATS = 40
SENS_KERNELS = {"Matern 5/2": 2.5, "Matern 3/2": 1.5}
SENS_ALPHAS = (1e-6, 1e-4, 1e-2)


def load():
    rows = S.load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows], float)
    y = {"D_t": np.array([r["Dt_pct"] for r in rows]),
         "log10_Q": np.log10([r["Q_uLs"] for r in rows]),
         "log10_tau_w": np.log10([r["tau_w_Pa"] for r in rows])}
    return rows, X, y


def folds(X, scheme):
    """Yield (group, held-out mask, 'interior' | 'edge') for one holdout scheme."""
    if scheme == "case":
        for i in range(len(X)):
            m = np.zeros(len(X), bool)
            m[i] = True
            yield i, m, "case"
        return
    col = {"diameter": 1, "pressure": 0}[scheme]
    levels = np.unique(X[:, col])
    for v in levels:
        yield float(v), X[:, col] == v, ("edge" if v in (levels[0], levels[-1]) else "interior")


def held_out(X, y, scheme, fitter, want_std=True):
    """Predictions for every case from the fold that held it out."""
    group, kind = [None] * len(y), [None] * len(y)
    pred = np.empty(len(y))
    std = np.full(len(y), np.nan)
    for g, te, k in folds(X, scheme):
        model = fitter(X[~te], y[~te])
        if want_std and hasattr(model, "gp"):
            mu, sd = model.predict(X[te], return_std=True)
            pred[te], std[te] = mu, sd
        else:
            pred[te] = model.predict(X[te])
        for i in np.flatnonzero(te):
            group[i], kind[i] = g, k
    return pred, std, group, kind


def scores(y, pred):
    e = pred - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "max_abs": float(np.max(np.abs(e)))}


def coverage(y, pred, std):
    inside = np.abs(pred - y) <= Z95 * std
    return {"n_inside": int(inside.sum()), "n": int(len(y)), "empirical": float(inside.mean()),
            "mean_width": float(np.mean(2 * Z95 * std))}


def decisions(dt_true, q_true, dt_pred, q_pred):
    ok_true = (dt_true < S.DT_MAX) & (q_true >= S.Q_MIN)
    ok_pred = (dt_pred < S.DT_MAX) & (q_pred >= S.Q_MIN)
    return {"accepted_by_fem": int(ok_true.sum()), "accepted_by_surrogate": int(ok_pred.sum()),
            "wrongly_accepted": int((ok_pred & ~ok_true).sum()),
            "wrongly_rejected": int((~ok_pred & ok_true).sum())}


class Poly:
    """Least-squares polynomial surface in (pressure, diameter), refitted on every fold."""

    def __init__(self, X, y, deg):
        self.deg = deg
        self.coef = np.linalg.lstsq(self._design(X), y, rcond=None)[0]

    def _design(self, X):
        p, d = X[:, 0], X[:, 1]
        cols = [np.ones_like(p)]
        for total in range(1, self.deg + 1):
            for i in range(total + 1):
                cols.append(p ** (total - i) * d ** i)
        return np.column_stack(cols)

    def predict(self, X):
        return self._design(X) @ self.coef


def gp_fitter(nu=2.5, alpha=S.ALPHA, log=None):
    kernel = S.KERNEL if (nu == 2.5) else ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0], nu=nu, length_scale_bounds=(1e-2, 1e2))

    def fit(Xtr, ytr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = S.FoldModel(Xtr, ytr, kernel=kernel, alpha=alpha)
        if log is not None:
            log.extend(str(x.message).splitlines()[0] for x in w
                       if issubclass(x.category, ConvergenceWarning))
        return m
    return fit


def at_bounds(kernel_):
    """Names of fitted hyperparameters that sit on a bound of their search range."""
    hits = []
    for hp in kernel_.hyperparameters:
        val = np.atleast_1d(kernel_.get_params()[hp.name])
        lo, hi = np.log(hp.bounds[:, 0]), np.log(hp.bounds[:, 1])
        v = np.log(val)
        if np.any(np.isclose(v, lo, atol=1e-3)) or np.any(np.isclose(v, hi, atol=1e-3)):
            hits.append(hp.name)
    return hits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X, y = load()
    R = {"grid": S.GRID.name, "n_cases": len(rows),
         "diameters_um": sorted({float(d) for d in X[:, 1]}),
         "pressures_kPa": sorted({float(p) for p in X[:, 0]}),
         "model": {"kernel": str(S.KERNEL), "alpha": S.ALPHA, "restarts": S.N_RESTARTS,
                   "seed": S.SEED, "normalize_y": True, "inputs": "z-scored within each fit",
                   "targets": list(y)}}
    print(f"  grid {S.GRID.name}: {len(rows)} cases\n")

    R["final_fit"] = {}
    for name, yy in y.items():
        log = []
        m = gp_fitter(log=log)(X, yy)
        R["final_fit"][name] = {"kernel": str(m.gp.kernel_),
                                "log_marginal_likelihood": float(m.gp.log_marginal_likelihood_value_),
                                "at_bounds": at_bounds(m.gp.kernel_),
                                "optimizer_warnings": sorted(set(log))}
        print(f"  {name:12s} {m.gp.kernel_}   at bounds: {at_bounds(m.gp.kernel_) or 'none'}"
              f"   warnings: {len(log)}")
    R["bounds"] = {"constant": [1e-3, 1e3], "length_scale": [1e-2, 1e2]}
    print()

    fold_rows, R["holdout"] = [], {}
    q_true = 10 ** y["log10_Q"]
    for scheme in ("case", "diameter", "pressure"):
        warn = []
        res = {k: held_out(X, yy, scheme, gp_fitter(log=warn)) for k, yy in y.items()}
        dt_p, dt_s, group, kind = res["D_t"]
        lq_p, lq_s, _, _ = res["log10_Q"]
        q_p = 10 ** lq_p
        block = {"D_t_pp": scores(y["D_t"], dt_p),
                 "log10_Q_dex": scores(y["log10_Q"], lq_p),
                 "log10_tau_w_dex": scores(y["log10_tau_w"], res["log10_tau_w"][0]),
                 "Q_abs_pct_error": {"mean": float(np.mean(np.abs(q_p / q_true - 1)) * 100),
                                     "max": float(np.max(np.abs(q_p / q_true - 1)) * 100)},
                 "coverage95_D_t": coverage(y["D_t"], dt_p, dt_s),
                 "coverage95_log10_Q": coverage(y["log10_Q"], lq_p, lq_s),
                 "decisions": decisions(y["D_t"], q_true, dt_p, q_p),
                 "optimizer_warnings": len(warn)}
        if scheme != "case":
            kinds = np.array(kind)
            per = {}
            for g in sorted(set(group)):
                m = np.array([gg == g for gg in group])
                per[str(g)] = {"kind": kinds[m][0],
                               "D_t_mae_pp": float(np.mean(np.abs(dt_p - y["D_t"])[m])),
                               "D_t_max_pp": float(np.max(np.abs(dt_p - y["D_t"])[m])),
                               "log10_Q_mae_dex": float(np.mean(np.abs(lq_p - y["log10_Q"])[m])),
                               "log10_Q_max_dex": float(np.max(np.abs(lq_p - y["log10_Q"])[m])),
                               "n": int(m.sum()),
                               "D_t_inside95": int((np.abs(dt_p - y["D_t"]) <= Z95 * dt_s)[m].sum()),
                               "log10_Q_inside95": int((np.abs(lq_p - y["log10_Q"]) <= Z95 * lq_s)[m].sum())}
            block["per_group"] = per
            for k in ("interior", "edge"):
                m = kinds == k
                block[f"{k}_D_t_pp"] = scores(y["D_t"][m], dt_p[m])
                block[f"{k}_log10_Q_dex"] = scores(y["log10_Q"][m], lq_p[m])
                block[f"{k}_coverage95_D_t"] = coverage(y["D_t"][m], dt_p[m], dt_s[m])
                block[f"{k}_coverage95_log10_Q"] = coverage(y["log10_Q"][m], lq_p[m], lq_s[m])
        block["baselines_D_t_pp"] = {}
        for deg, nm in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
            bp = held_out(X, y["D_t"], scheme, lambda a, b, d=deg: Poly(a, b, d), False)[0]
            block["baselines_D_t_pp"][nm] = scores(y["D_t"], bp)
        R["holdout"][scheme] = block
        for i in range(len(y["D_t"])):
            fold_rows.append({"holdout": scheme, "group": group[i], "kind": kind[i],
                              "needle_um": X[i, 1], "pressure_kPa": X[i, 0],
                              "Dt_fem_pct": y["D_t"][i], "Dt_pred_pct": dt_p[i],
                              "Dt_sd_pp": dt_s[i], "Q_fem_uLs": q_true[i], "Q_pred_uLs": q_p[i],
                              "log10_Q_sd": lq_s[i]})
        b = block
        print(f"  {scheme:9s} D_t MAE {b['D_t_pp']['mae']:.3f}  RMSE {b['D_t_pp']['rmse']:.3f}  "
              f"max {b['D_t_pp']['max_abs']:.3f} pp | Q MAE {b['log10_Q_dex']['mae']:.4f} dex, "
              f"{b['Q_abs_pct_error']['mean']:.1f} % | cover D_t "
              f"{b['coverage95_D_t']['n_inside']}/{b['coverage95_D_t']['n']}, Q "
              f"{b['coverage95_log10_Q']['n_inside']}/{b['coverage95_log10_Q']['n']} | "
              f"wrong accept {b['decisions']['wrongly_accepted']}, "
              f"wrong reject {b['decisions']['wrongly_rejected']}")
        if scheme != "case":
            print(f"            interior D_t {b['interior_D_t_pp']['mae']:.3f} pp, "
                  f"edge {b['edge_D_t_pp']['mae']:.3f} pp")
    table = [{"holdout": sch, "held_out": g, **v} for sch in ("diameter", "pressure")
             for g, v in R["holdout"][sch]["per_group"].items()]
    with open(OUT / "blocked_holdouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "fold_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        w.writeheader()
        w.writerows(fold_rows)
    print()

    sens = []
    for kname, nu in SENS_KERNELS.items():
        for alpha in SENS_ALPHAS:
            fit = gp_fitter(nu, alpha)
            row = {"kernel": kname, "alpha": alpha}
            for scheme in ("case", "diameter", "pressure"):
                dp, ds, _, _ = held_out(X, y["D_t"], scheme, fit)
                lp, ls, _, _ = held_out(X, y["log10_Q"], scheme, fit)
                row[f"{scheme}_D_t_mae_pp"] = scores(y["D_t"], dp)["mae"]
                row[f"{scheme}_log10_Q_mae_dex"] = scores(y["log10_Q"], lp)["mae"]
                row[f"{scheme}_cover_D_t"] = coverage(y["D_t"], dp, ds)["empirical"]
                row[f"{scheme}_cover_log10_Q"] = coverage(y["log10_Q"], lp, ls)["empirical"]
            m_dt, m_q = fit(X, y["D_t"]), fit(X, y["log10_Q"])
            z = lambda P, D: np.column_stack([np.ravel(P), np.ravel(D)])  # noqa: E731
            for d in (100.0, 200.0, 250.0, 330.0):
                op = S.operating_point(m_dt, m_q, z, d)
                row[f"P_opt_{d:.0f}_kPa"] = None if op is None else round(op["P_kPa"], 1)
            sens.append(row)
            print(f"  {kname}  alpha {alpha:.0e}   case {row['case_D_t_mae_pp']:.3f}  "
                  f"diameter {row['diameter_D_t_mae_pp']:.3f}  pressure "
                  f"{row['pressure_D_t_mae_pp']:.3f} pp   cover(case) {row['case_cover_D_t']:.3f}"
                  f"   P_opt {[row[f'P_opt_{d}_kPa'] for d in (100, 200, 250, 330)]}")
    with open(OUT / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0]))
        w.writeheader()
        w.writerows(sens)
    R["sensitivity"] = sens
    print()

    rng = np.random.default_rng(S.SEED)
    curve = []
    fit = gp_fitter()
    # The sizes and draws of the learning-curve check: every third size from 6 up to two short of the grid.
    for n in range(6, len(X) - 1, 3):
        errs = []
        for _ in range(LEARNING_REPEATS):
            idx = rng.permutation(len(X))
            tr, te = idx[:n], idx[n:]
            errs.append(float(np.mean(np.abs(fit(X[tr], y["D_t"][tr]).predict(X[te]) - y["D_t"][te]))))
        curve.append({"n_train": n, "mae_pp": float(np.mean(errs)),
                      "sd_across_random_splits_pp": float(np.std(errs))})
    R["learning_curve"] = curve

    m_dt, m_q = fit(X, y["D_t"]), fit(X, y["log10_Q"])
    z = lambda P, D: np.column_stack([np.ravel(P), np.ravel(D)])  # noqa: E731
    ops = []
    for d in R["diameters_um"]:
        op = S.operating_point(m_dt, m_q, z, d)
        ops.append(op or {"needle_um": d, "P_kPa": None, "Dt_pct": None, "Q_uLs": None,
                          "note": "no feasible condition"})
    opt_dir = OUT.parent / "optimal_params"
    opt_dir.mkdir(exist_ok=True)
    with open(opt_dir / "L8_operating_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["needle_um", "P_kPa", "Dt_pct", "Q_uLs", "note"])
        w.writeheader()
        for o in ops:
            w.writerow({k: o.get(k, "") for k in w.fieldnames})
    R["operating_points"] = ops

    fits = []
    for _ in range(5):                   # median of repeats: a single fit time is noisy
        t0 = time.perf_counter()
        for yy in y.values():
            S.FoldModel(X, yy)
        fits.append(time.perf_counter() - t0)
    fit_s = float(np.median(fits))
    one = np.array([[200.0, 250.0]])
    single = []
    for _ in range(2000):
        t = time.perf_counter()
        m_dt.predict(one)
        single.append(time.perf_counter() - t)
    sweep = np.column_stack([np.tile(np.linspace(*S.P_TRAINED, 3751), 4),
                             np.repeat(R["diameters_um"], 3751)])
    batch = []
    for _ in range(50):
        t = time.perf_counter()
        m_dt.predict(sweep)
        m_q.predict(sweep)
        batch.append(time.perf_counter() - t)
    R["timing"] = {
        "hardware": {"machine": platform.machine(), "processor": _cpu_model(),
                     "logical_cpus": os.cpu_count(), "system": platform.platform()},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "fit_three_gps_s": fit_s, "fit_repeats": len(fits),
        "single_query_ms": {"median": 1e3 * float(np.median(single)),
                            "p95": 1e3 * float(np.percentile(single, 95)), "repeats": len(single)},
        "operating_window_sweep": {"points": int(len(sweep)), "outputs": 2,
                                   "median_s": float(np.median(batch)), "repeats": len(batch)},
        "fem_campaign": fem_campaign(len(rows))}
    print(f"  fit 3 GPs {fit_s:.2f} s (median of {len(fits)}); single query {R['timing']['single_query_ms']['median']:.3f} ms "
          f"(median of 2000); sweep of {len(sweep)} points x 2 outputs "
          f"{R['timing']['operating_window_sweep']['median_s'] * 1e3:.1f} ms")

    (OUT / "surrogate_validation.json").write_text(json.dumps(R, indent=1))
    print(f"\n  written {OUT.relative_to(OUT.parents[1])}/: surrogate_validation.json, "
          f"fold_predictions.csv, sensitivity.csv; results/optimal_params/L8_operating_points.csv")


TIMING = pathlib.Path(__file__).resolve().parents[1] / "results" / "fem" / "runs" / \
    "2026-08-26_ning-grid32" / "timing_M12.csv"


def fem_campaign(n_cases):
    """CPU time of the canonical campaign, from its job logs (fem_campaign_timing.py)."""
    h = [float(r["cpu_hours"]) for r in csv.DictReader(open(TIMING))]
    if len(h) != n_cases:
        raise SystemExit(f"{TIMING.name} has {len(h)} cases, the grid {n_cases}")
    return {"cases": len(h), "source": f"{TIMING.parent.name}/{TIMING.name}",
            "hours_per_case_mean": float(np.mean(h)), "hours_per_case_median": float(np.median(h)),
            "hours_min": min(h), "hours_max": max(h), "cpu_hours": float(sum(h))}


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
