"""Surrogate diagnostics; not a figure of the article.

Leave-one-out parity, learning curve, comparison with polynomial surfaces and interval
calibration of the total-damage surrogate, on one sheet. figS3_interval_calibration.py draws the
calibration panel alone.

    python scripts/surrogate_diagnostics.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l8_style as S
import l8_surrogate as SUR
from l8_paths import save_figure

warnings.filterwarnings("ignore")

FIGNAME = "surrogate_diagnostics.png"
# Training sizes for the learning curve, derived from the grid at run time so the curve
# always runs up to the full campaign it is meant to justify.
CURVE_N: tuple[int, ...] = ()   # filled in main() from the loaded grid
CURVE_DRAWS = 40
CURVE_SEED = 0


class _GP:
    """The surrogate as a fit/predict object: SUR.FoldModel refits the input scaling and
    the hyperparameters on whatever rows it is given, exactly as surrogate_validation.py does."""

    def fit(self, X, y):
        self.m = SUR.FoldModel(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.m.predict(X, return_std=return_std)


gp = _GP


def leave_one_out(Xz, y, make, with_sd=False):
    """Refit on n-1, predict the held-out 1, for every case in the grid.

    with_sd also returns the posterior standard deviation of each held-out prediction,
    which is what panel (d) needs: an uncertainty is only worth printing if the truth
    actually falls inside it at about the stated rate.
    """
    out = np.empty(len(Xz))
    sd = np.empty(len(Xz))
    for i in range(len(Xz)):
        m = np.ones(len(Xz), bool)
        m[i] = False
        g = make()
        g.fit(Xz[m], y[m])
        if with_sd:
            mu, s = g.predict(Xz[i:i + 1], return_std=True)
            out[i], sd[i] = mu[0], s[0]
        else:
            out[i] = g.predict(Xz[i:i + 1])[0]
    return (out, sd) if with_sd else out


def learning_curve(Xz, y):
    rng = np.random.default_rng(CURVE_SEED)
    mean, sd = [], []
    for n in CURVE_N:
        errs = []
        for _ in range(CURVE_DRAWS):
            idx = rng.permutation(len(Xz))
            tr, te = idx[:n], idx[n:]
            g = gp()
            g.fit(Xz[tr], y[tr])
            errs.append(np.mean(np.abs(g.predict(Xz[te]) - y[te])))
        errs = np.asarray(errs)
        mean.append(errs.mean())
        sd.append(errs.std())
    return np.asarray(mean), np.asarray(sd)


def panel_parity(ax, y, pred, dia, mae, r2):
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color=S.THRESHOLD, lw=1.6,
            zorder=1)
    for d in sorted(set(dia)):
        m = dia == d
        ax.plot(y[m], pred[m], marker=S.dia_marker(d), color=S.dia_color(d),
                label=S.dia_label(d),
                markeredgecolor="white", markersize=S.PAPER["ms"], linestyle="none",
                zorder=3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    # A parity plot is read by going across and down, so the identity line has to sit at 45 degrees.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("FEM  $D_t$  (%)")
    ax.set_ylabel("Surrogate, held out (%)")
    # The panel is named by its letter alone; titles live in the manuscript.
    S.panel_label(ax, "A")
    ax.legend(loc="upper left", title="Nozzle diameter", fontsize=S.PAPER["legend"] - 1.5)
    # The result value belongs in the caption, not on the axes; it is printed to stdout so the
    # caption stays in step with the data.
    print(f"  caption (a)  ·  MAE {mae:.3f} pp, R2 = {r2:.4f}")


def panel_curve(ax, mean, sd, full_mae, n_total):
    n = np.asarray(CURVE_N)
    ax.fill_between(n, np.maximum(mean - sd, 0), mean + sd, color=S.dia_color(200),
                    alpha=0.30, zorder=2, linewidth=0)
    ax.plot(n, mean, "o-", color=S.dia_color(330), markeredgecolor="white",
            lw=S.PAPER["lw_data"], zorder=3)
    # Named in a key. The line is the error the surrogate reaches once every run is available,
    # i.e. the floor the curve is approaching, and the key gives the coloured rule its meaning.
    ax.axhline(full_mae, color=S.ANNOT, ls="--", lw=1.8, zorder=4,
               label=f"all {n_total} runs")
    ax.legend(loc="upper right", frameon=False, fontsize=S.PAPER["legend"] - 0.5,
              handlelength=1.6, borderaxespad=0.4)
    # The result value belongs in the caption, not on the axes; it is printed to stdout so the
    # caption stays in step with the data.
    print(f"  caption (b)  ·  all {n_total} runs: {full_mae:.3f} pp")
    ax.set_ylim(0, None)
    ax.set_xlabel("FEM runs used for training")
    ax.set_ylabel("Held-out MAE (pp)")
    S.panel_label(ax, "B")
    ax.set_xticks(list(n))


def panel_models(ax, scores):
    names = [k for k, _ in scores]
    vals = [v for _, v in scores]
    ypos = np.arange(len(names))[::-1]
    colors = [S.ANNOT if k.startswith("Gaussian") else S.COMP_C["shear"] for k in names]
    edge = [S.INK] * len(names)
    ax.barh(ypos, vals, height=0.6, color=colors, edgecolor=edge, linewidth=1.0, zorder=3)
    # The result value belongs in the caption, not on the axes; it is printed to stdout so the
    # caption stays in step with the data.
    print("  caption (c)  ·  " + "  ".join(f"{nm}: {v:.3f} pp"
                                            for nm, v in zip(names, vals)))
    # The names go to the right of their own bars.
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, float(ypos.max()) + 0.6)
    pad = max(vals) * 0.025
    labels = [ax.text(v + pad, yy, nm, ha="left", va="center",
                      fontsize=S.PAPER["tick"], color=S.INK, zorder=4)
              for yy, nm, v in zip(ypos, names, vals)]

    # The limit is set from the labels, not guessed.
    ax.set_xlim(0, max(vals) * 1.32)
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    need = 0.0
    for t, v in zip(labels, vals):
        w_px = t.get_window_extent(rend).width
        x0 = ax.transData.transform((0, 0))[0]
        w_data = ax.transData.inverted().transform((x0 + w_px, 0))[0]
        need = max(need, v + pad + w_data)
    if need > ax.get_xlim()[1]:
        ax.set_xlim(0, need * 1.03)
    print(f"  panel (c): labels need x up to {need:.2f}; axis set to {ax.get_xlim()[1]:.2f}")

    ax.set_xlabel("Leave-one-out MAE (pp)")
    S.panel_label(ax, "C")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)


def panel_calibration(ax, y, pred, sd, dia, letter="D"):
    """Held-out error against the interval the surrogate claimed for that prediction.

    A point below the line is a case whose true value fell inside its own nominal 95 %
    interval. This is the panel that decides whether the posterior standard deviation
    drawn on the response surface can be read as an error bar, and the answer here is
    that it cannot: 27 of 32 fall inside, and with 32 dependent predictions the size of the
    undercoverage is itself uncertain. Drawn alone as Additional file 1, Fig. S3
    (figS3_interval_calibration.py), where `letter` is None.
    """
    half = 1.96 * sd
    err = np.abs(pred - y)
    inside = int(np.sum(err <= half))
    hi = max(half.max(), err.max()) * 1.10
    ax.plot([0, hi], [0, hi], "-", color=S.THRESHOLD, lw=1.6, zorder=1)
    for d in sorted(set(dia)):
        m = dia == d
        ax.plot(half[m], err[m], marker=S.dia_marker(d), color=S.dia_color(d),
                markeredgecolor="white",
                markersize=S.PAPER["ms"], linestyle="none", zorder=3)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    # "datalim", not "box".
    ax.set_aspect("equal", adjustable="datalim")
    # The 1.96 sigma definition is in the caption.
    ax.set_xlabel("Nominal 95 % half-width (pp)")
    ax.set_ylabel("Actual held-out error (pp)")
    if letter:
        S.panel_label(ax, letter)
    # The result value belongs in the caption, not on the axes; it is printed to stdout so the
    # caption stays in step with the data.
    print(f"  caption (d)  ·  {inside} of {len(y)} inside "
          f"({100 * inside / len(y):.0f} % vs 95 % nominal)")
    return inside


def main():
    rows = SUR.load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    y = np.array([r["Dt_pct"] for r in rows])
    dia = np.array([int(r["needle_um"]) for r in rows])
    Xz = (X - X.mean(0)) / X.std(0)

    global CURVE_N
    n_total = len(rows)
    # Up to n_total - 2, so at least two cases are always held out to score against
    CURVE_N = tuple(range(6, n_total - 1, 3))

    pred, pred_sd = leave_one_out(X, y, gp, with_sd=True)
    mae = float(np.mean(np.abs(pred - y)))
    r2 = 1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  leave-one-out: MAE {mae:.3f} pp, R2 {r2:.4f}")
    if abs(mae - SUR.LOO_DT_REPORTED) > SUR.LOO_TOL:
        raise SystemExit(f"REFUSING: LOO MAE {mae:.3f} pp is not the reported "
                         f"{SUR.LOO_DT_REPORTED:.3f} pp")

    scores = [("Gaussian process", mae)]
    # Ascending polynomial degree, after the reference model, so the axis reads in the
    # order a reader expects.
    for name, make in (("linear", LinearRegression),
                       ("quadratic", lambda: make_pipeline(PolynomialFeatures(2),
                                                           LinearRegression())),
                       ("cubic", lambda: make_pipeline(PolynomialFeatures(3),
                                                       LinearRegression()))):
        e = float(np.mean(np.abs(leave_one_out(Xz, y, make) - y)))
        scores.append((name, e))
        print(f"  {name:>16s}: MAE {e:.3f} pp")

    mean, sd = learning_curve(X, y)
    print(f"  learning curve: {mean[0]:.2f} pp at n={CURVE_N[0]} "
          f"-> {mean[-1]:.2f} pp at n={CURVE_N[-1]}")

    with plt.rc_context(S.rc(S.PAPER)):
        # Four panels in a row would be about 40 mm each and the parity and calibration boxes are
        # square, so a 2 x 2 grid is what fits.
        fig, ax = plt.subplots(2, 2, figsize=(6.9191, 6.1481))
        panel_parity(ax[0, 0], y, pred, dia, mae, r2)
        panel_curve(ax[0, 1], mean, sd, mae, n_total)
        panel_models(ax[1, 0], scores)
        inside = panel_calibration(ax[1, 1], y, pred, pred_sd, dia)
        print(f"  interval coverage: {inside}/{n_total} inside the nominal 95 % band "
              f"({100 * inside / n_total:.1f} %)")
        fig.tight_layout(pad=1.1, w_pad=3.0, h_pad=2.0)
        # Panel letters snap to each panel's real top-left corner once the layout is fixed.
        S.snap_panel_labels(fig)
        # No two panel titles may touch. Measured from the renderer, because overprinted titles
        # interleave their glyphs, and short strings are not a guarantee on their own; the check is.
        fig.canvas.draw()
        _r = fig.canvas.get_renderer()
        _t = [a.title for a in ax.ravel() if a.title.get_text().strip()]
        for _i in range(len(_t)):
            for _j in range(_i + 1, len(_t)):
                _a = _t[_i].get_window_extent(_r); _b = _t[_j].get_window_extent(_r)
                _dx = min(_a.x1, _b.x1) - max(_a.x0, _b.x0)
                _dy = min(_a.y1, _b.y1) - max(_a.y0, _b.y0)
                assert not (_dx > 0 and _dy > 0), (
                    f"panel titles overlap: {_t[_i].get_text()!r} x {_t[_j].get_text()!r}")
        print("  panel-title collision check: none of "
              f"{len(_t)} titles overlap")
        for w in save_figure(fig, FIGNAME, S.PAPER["dpi"], tight=True):
            print(f"  wrote {w}")


if __name__ == "__main__":
    main()
