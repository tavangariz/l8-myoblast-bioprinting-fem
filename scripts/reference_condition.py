#!/usr/bin/env python3
"""Three checks at the reference condition (200 kPa, 200 µm nozzle), from the solved field.

1. Wall friction in the cartridge and the conical transition, integrated from the solved wall shear
   stress, and its share of the entrance pressure (Results, Damage decomposition at the reference
   condition).
2. The entrance pressure and extensional damage if the wall shear stress were taken from the finite
   element field instead of the Rabinowitsch relation (Discussion).
3. The shear damage the cartridge and the transition would add if the shear law were applied there
   (Limitations).

    python scripts/reference_condition.py
"""
from __future__ import annotations

import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8_damage as D  # noqa: E402
from l8_paths import CANONICAL_GRID, CHECKPOINT, require  # noqa: E402


def shear_damage(tau: float, t: float) -> float:
    """Local shear damage of Ning et al. (2018), Eq. 20."""
    return 1.0 - math.exp(-D.A1 * tau ** D.B1 * t ** D.C1)


def main() -> int:
    ck = pickle.load(open(require(CHECKPOINT, "the solver checkpoint"), "rb"))
    xy = np.asarray(ck["vtk_coords"])
    tau = np.asarray(ck["fields"]["tau"]).ravel()
    r, z = xy[:, 0], xy[:, 1]
    Rin, Rmid, Rtip = ck["Rin"], ck["Rmid"], ck["Rtip"]
    z1, z2, z3 = ck["z1"], ck["z2"], ck["z3"]
    q_uls = ck["Q_uLs"]
    Q = q_uls * 1e-9

    def r_wall(zz):
        return np.where(zz <= z1, Rin, np.where(zz < z2, Rin + (Rmid - Rin) * (zz - z1) / (z2 - z1), Rtip))

    def wall_drop(zlo, zhi, nb):
        """Pressure drop 2*tau_w/R integrated along z, tau_w from the outermost node of each axial slice."""
        edges = np.linspace(zlo, zhi, nb + 1)
        zc, tw = [], []
        for i in range(nb):
            m = (z >= edges[i]) & (z < edges[i + 1])
            if not m.any():
                continue
            zm = 0.5 * (edges[i] + edges[i + 1])
            idx = np.where(m)[0]
            j = idx[np.argmax(r[idx])]
            if r[j] < 0.95 * float(r_wall(np.array([zm]))[0]):
                continue
            zc.append(zm)
            tw.append(tau[j])
        zc, tw = np.array(zc), np.array(tw)
        zf = np.concatenate([[zlo], zc, [zhi]])
        tf = np.concatenate([[tw[0]], tw, [tw[-1]]])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz       # NumPy 2 / NumPy 1
        return float(trapezoid(2 * tf / r_wall(zf), zf)), float(tw.mean())

    grid = {(int(float(g["needle_um"])), float(g["pressure_kPa"])): g
            for g in csv.DictReader(open(require(CANONICAL_GRID, "the 32-case grid")))}
    ref = grid[(200, 200.0)]
    p_en = float(ref["P_en_kPa"])

    # 1. wall friction upstream of the nozzle
    drops = [wall_drop(0.0, z1, nb)[0] for nb in (12, 20, 30)]
    dp_cart = float(np.mean(drops))
    dp_tr, tw_tr = wall_drop(z1, z2, 20)
    feed = (dp_cart + dp_tr) / 1e3
    print(f"1. wall friction: cartridge {dp_cart / 1e3:.4f} kPa, transition {dp_tr / 1e3:.4f} kPa, "
          f"total {feed:.3f} kPa = {100 * feed / p_en:.1f} % of the entrance pressure ({p_en:.3f} kPa)")

    # 2. closure sensitivity: wall shear stress from the field instead of the Rabinowitsch relation
    tw_an, tw_fem = ck["tau_w"], ck["tau_w_fem"]
    pen_an = 200.0 - 2 * tw_an * (z3 - z2) / Rtip / 1e3
    pen_fem = 200.0 - 2 * tw_fem * (z3 - z2) / Rtip / 1e3
    det = lambda pen: 100 * D.compute_Det(3 * (D.N_EXP + 1) / 8 * pen * 1e3)     # Eqs. 22-23
    print(f"2. wall shear stress {tw_an:.1f} Pa (analytic) and {tw_fem:.1f} Pa (field), "
          f"{100 * abs(tw_fem / tw_an - 1):.2f} % apart: entrance pressure {pen_an:.2f} -> {pen_fem:.2f} kPa, "
          f"Det {det(pen_an):.2f} -> {det(pen_fem):.2f} %")

    # 3. shear exposure upstream of the nozzle
    v_cart = math.pi * Rin ** 2 * z1
    v_tr = math.pi * (z2 - z1) / 3 * (Rin ** 2 + Rin * Rmid + Rmid ** 2)
    v_tip = math.pi * Rtip ** 2 * (z3 - z2)
    t_cart, t_tr, t_tip = v_cart / Q, v_tr / Q, v_tip / Q
    tw_cart = D.tau_w_analytic(q_uls, Rin)
    d_cart, d_tr = shear_damage(tw_cart, t_cart), shear_damage(tw_tr, t_tr)
    d_tip = float(ref["Dst_pct"]) / 100
    cascade = 1 - (1 - d_cart) * (1 - d_tr) * (1 - d_tip)
    print(f"3. residence time: cartridge {t_cart:.2f} s, transition {t_tr:.2f} s, nozzle {t_tip:.3f} s; "
          f"the shear law at {tw_cart:.1f} Pa (cartridge) and {tw_tr:.1f} Pa (transition) would add "
          f"{100 * (cascade - d_tip):.2f} percentage points to Dst ({100 * d_tip:.2f} in the nozzle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
