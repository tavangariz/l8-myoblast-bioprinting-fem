"""The Gaussian process surrogate.

One GP each for total damage, log10 Q and log10 tau_w, with applied pressure and nozzle diameter
as inputs: Matérn 5/2 kernel times a constant, standardized inputs, normalized targets,
alpha = 1e-6, ten optimizer restarts, seed 0. Every figure and validation script imports the
surrogate from here. fit_all() repeats the leave-one-out cross-validation and stops if the mean
absolute error in total damage differs from the 0.338 percentage points reported in the article.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from l8_paths import CANONICAL_GRID, require

GRID = require(CANONICAL_GRID, "the canonical FEM grid")

# The configuration of the article. Changing anything here changes every figure that shows the surrogate.
KERNEL = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
    length_scale=[1.0, 1.0], nu=2.5, length_scale_bounds=(1e-2, 1e2))
ALPHA = 1e-6
N_RESTARTS = 10
SEED = 0

LOO_DT_REPORTED = 0.338        # pp, leave-one-case-out on the 32-case grid, reported in the article.
                              # A version-drift guard, not an accuracy requirement: it stops
                              # every importer if the grid, the configuration or a library
                              # changes the fit. The accuracy evidence is in
                              # surrogate_validation.py.
LOO_TOL = 0.005


def _maybe_float(v):
    """Numeric columns become floats; provenance columns stay text."""
    try:
        return float(v)
    except ValueError:
        return v


def load_grid(path: Path = GRID) -> list[dict]:
    """The canonical grid: four tip diameters by eight applied pressures.

    The row count is checked against the diameters and pressures the file itself contains,
    rather than against a constant, so the check follows the grid when it is widened.
    """
    rows = [{k: _maybe_float(v) for k, v in r.items() if v != ""}
            for r in csv.DictReader(open(path))]
    dias = {round(r["needle_um"]) for r in rows}
    press = {r["pressure_kPa"] for r in rows}
    if len(rows) != len(dias) * len(press):
        raise SystemExit(f"REFUSING: {len(rows)} rows is not a full "
                         f"{len(dias)} x {len(press)} grid")
    return rows


def _fit(Xz, y, restarts=N_RESTARTS, kernel=KERNEL, alpha=ALPHA):
    g = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                 n_restarts_optimizer=restarts, random_state=SEED)
    g.fit(Xz, y)
    return g


class FoldModel:
    """The surrogate as it is fitted inside one cross-validation fold.

    The input standardisation is fitted on the training rows of the fold only, and the
    kernel hyperparameters are re-optimised there too, so nothing about a held-out case
    reaches the model that predicts it. On the full grid this is exactly the deployment
    fit of `fit_all`.
    """

    def __init__(self, X, y, restarts=N_RESTARTS, kernel=KERNEL, alpha=ALPHA):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.gp = _fit((X - self.mu) / self.sd, y, restarts, kernel, alpha)

    def predict(self, X, return_std=False):
        return self.gp.predict((np.asarray(X, float) - self.mu) / self.sd,
                               return_std=return_std)


def fit_all(rows: list[dict] | None = None, check: bool = True):
    """Fit the surrogate on (pressure, diameter) -> D_t and log10 Q.

    Returns (gp_dt, gp_logq, z) where z(P, D) maps physical inputs to the fitted space.
    """
    rows = rows or load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    Xz = (X - mu) / sd

    dt = np.array([r["Dt_pct"] for r in rows])
    logq = np.log10([r["Q_uLs"] for r in rows])

    gp_dt, gp_logq = _fit(Xz, dt), _fit(Xz, logq)

    if check:
        errs = []
        for i in range(len(X)):
            m = np.ones(len(X), bool)
            m[i] = False
            errs.append(abs(FoldModel(X[m], dt[m]).predict(X[i:i + 1])[0] - dt[i]))
        loo = float(np.mean(errs))
        if abs(loo - LOO_DT_REPORTED) > LOO_TOL:
            raise SystemExit(
                f"REFUSING: LOO-CV mean |ΔD_t| is {loo:.3f} pp, the article reports "
                f"{LOO_DT_REPORTED:.3f} pp. The surrogate or the grid has moved.")
        print(f"  surrogate verified: LOO-CV mean |ΔD_t| = {loo:.3f} pp over {len(rows)} folds")

    def z(P, D):
        return (np.column_stack([np.ravel(P), np.ravel(D)]) - mu) / sd

    return gp_dt, gp_logq, z


def fit_field(key: str, rows: list[dict] | None = None, log10: bool = False):
    """Fit the same surrogate to any column of the grid, for figures showing a field that
    `fit_all` does not return. Identical kernel, scaling and seed, so every surface in the
    paper comes from one configuration.

    Returns (gp, z) with the same meaning as in `fit_all`.
    """
    rows = rows or load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    y = np.array([r[key] for r in rows])
    if log10:
        y = np.log10(y)
    gp = _fit((X - mu) / sd, y)

    def z(P, D):
        return (np.column_stack([np.ravel(P), np.ravel(D)]) - mu) / sd

    return gp, z


# The operating-window criteria, and the domain the surrogate was trained on. Recommendations
# are made only inside that domain; a condition outside it is not a prediction of this model.
DT_MAX = 20.0                  # %, total damage must be strictly below this
Q_MIN = 1.0                    # µL/s, flow rate must reach this
P_TRAINED = (25.0, 400.0)      # kPa, the applied pressures of the canonical grid
D_TRAINED = (100.0, 330.0)     # µm, the tip diameters of the canonical grid


def operating_point(gp_dt, gp_logq, z, diameter_um: float, q_min: float = Q_MIN,
                    dt_max: float = DT_MAX, n: int = 3751):
    """Lowest predicted D_t over the trained pressure range with D_t < dt_max and Q >= q_min.

    Both criteria are enforced on the same sweep, so a minimum that breaks the damage limit
    is never reported. Returns None when no pressure in the trained range satisfies both,
    which callers report as "no feasible condition". Refuses a diameter outside the trained
    range rather than extrapolating.
    """
    if not D_TRAINED[0] <= diameter_um <= D_TRAINED[1]:
        raise ValueError(f"{diameter_um} µm is outside the trained diameters {D_TRAINED}")
    P = np.linspace(*P_TRAINED, n)
    Xz = z(P, np.full_like(P, diameter_um))
    dt = gp_dt.predict(Xz)
    q = 10 ** gp_logq.predict(Xz)
    ok = (dt < dt_max) & (q >= q_min)
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[np.argmin(dt[ok])]
    return {"needle_um": float(diameter_um), "P_kPa": float(P[i]),
            "Dt_pct": float(dt[i]), "Q_uLs": float(q[i])}
