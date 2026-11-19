#!/usr/bin/env python3
"""Additional file 2, Fig. S4: the apparent viscosity the simulations use, against shear rate.

Power law eta = K gamma_dot^(n - 1) with K = 25.60 Pa s^n and n = 0.35 (Ning et al. 2018), bounded between
0.003 and 500 Pa s as in the solver (Additional file 1, Eq. S7). The shaded band is the range of wall shear
rates the 32 conditions reach, from the Rabinowitsch relation (Eq. 16) at their solved flow rates.

    python scripts/figS4_viscosity.py
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import l8_style as S
import l8_surrogate as SUR
from l8_damage import K_VIS, N_EXP
from l8_paths import FIGDIR

ETA_MIN, ETA_MAX = 0.003, 500.0


def main():
    rows = SUR.load_grid()
    gw = [(3 * N_EXP + 1) / (4 * N_EXP) * 4 * r["Q_uLs"] * 1e-9 / (math.pi * (r["needle_um"] * 0.5e-6) ** 3) for r in rows]
    g = np.logspace(-4, 6, 400)
    eta = np.clip(K_VIS * g ** (N_EXP - 1), ETA_MIN, ETA_MAX)
    plt.rcParams.update(S.rc(S.PAPER))
    fig, ax = plt.subplots(figsize=(S.PAGE_W_IN * 0.62, 2.9), dpi=S.PAPER["dpi"])
    ax.axvspan(min(gw), max(gw), color="#DDDDDD", zorder=0, label="wall shear rates of the 32 conditions")
    ax.loglog(g, K_VIS * g ** (N_EXP - 1), color=S.INK, lw=0.9, ls="--", label=r"power law, $K\dot{\gamma}^{\,n-1}$")
    ax.loglog(g, eta, color=S.COMP_C["total"], lw=S.PAPER["lw_data"], label="as solved, with bounds")
    ax.set_xlabel(r"Shear rate  $\dot{\gamma}$  (s$^{-1}$)")
    ax.set_ylabel(r"Apparent viscosity  $\eta$  (Pa s)")
    ax.legend(loc="lower left", fontsize=S.PAPER["legend"] - 0.5, frameon=False)
    fig.tight_layout()
    out = FIGDIR / "FigS4_viscosity.png"
    fig.savefig(out, dpi=S.PAPER["dpi"])
    print(f"wall shear rates: {min(gw):.3g} to {max(gw):.3g} 1/s; wrote {out}")


if __name__ == "__main__":
    main()
