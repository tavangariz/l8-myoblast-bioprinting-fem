#!/usr/bin/env python3
"""Damage law of Ning et al. (2018), as applied in this study.

derive(P_kPa, Q_uLs, R) takes the applied pressure, the solved flow rate and the nozzle radius and
returns the wall shear stress (Rabinowitsch relation, Eq. 16), the entrance pressure (Eq. 17), the
effective extensional stress (Eq. 22), the shear damage averaged over the nozzle cross-section
(Eqs. 18-21), the extensional damage (Eq. 23) and the total damage (Eq. 24). Every damage value in
the article comes from this function; the module reads no files.

Reference condition, 200 µm nozzle at 200 kPa:

    derive(200.0, 17.940002925163476, 100e-6)
      tau_w    981.2178 Pa      dP_tip   196.2436 kPa
      P_en       3.7564 kPa     tau_ext 1901.6964 Pa
      Dst        0.2864 %       Det        8.6948 %      Dt   8.9563 %

Running the file checks this row.

Parameters, from Ning et al. (2018): K and n, consistency and flow-behavior index of the L8
suspension; RHO, its density; LTIP, nozzle length (10 mm); A1, B1, C1, shear-damage coefficients
fitted over 100-1000 Pa and 10-120 s; A2, B2, extensional-damage coefficients fitted over
440-9251 Pa.
"""
from __future__ import annotations

import math

import numpy as np

from l8_geometry import TIP_RADIUS_M  # noqa: F401  (re-exported for callers)

# Fluid and geometry. Unchanged for the life of the study.
K_VIS, N_EXP, RHO = 25.60, 0.35, 990.0
LTIP = 10e-3

# Ning 2018 damage law.
A1, B1, C1 = 3.20e-4, 0.61, 0.52
A2, B2 = 1.43e-3, 0.55

# The calibrated box the coefficients were fitted inside. Reported alongside every damage
# value so a reader can see which predictions interpolate and which extrapolate.
SHEAR_BOX_TAU_PA = (100.0, 1000.0)
SHEAR_BOX_TIME_S = (10.0, 120.0)
EXT_BOX_TAU_PA = (440.0, 9251.0)

# A pure-numpy Gauss-Legendre rule, so the damage chain depends on NumPy alone. 400 nodes on
# a smooth integrand is ample, and the self-test below reproduces the solver's own column.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)


def tau_w_analytic(Q_uLs: float, R: float) -> float:
    """Rabinowitsch wall shear stress from flow rate, Pa.

    Exact for developed power-law flow in a straight tube:

        gamma_dot_w = (3n+1)/(4n) * 4Q/(pi R^3),    tau_w = K gamma_dot_w^n
    """
    Q = Q_uLs * 1e-9
    gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * Q / (math.pi * R ** 3)
    return K_VIS * gdot ** N_EXP


def velocity_profile(dP_tip: float, R: float, r):
    """Axial velocity at radius `r` in the nozzle, m/s. Power-law, developed, no-slip."""
    C = (dP_tip / (2 * K_VIS * LTIP)) ** (1 / N_EXP)
    alpha = (N_EXP + 1) / N_EXP
    return (N_EXP / (N_EXP + 1)) * C * (R ** alpha - np.asarray(r, float) ** alpha)


def compute_Dst(dP_tip: float, R: float) -> float:
    """Area-averaged shear damage in the nozzle, Ning Eq. (9).

        D_st,A = (2/R^2) * integral_0^R r * D_s(r) dr

    The axial velocity vanishes at the wall, so the residence time t_s = L/v_z(r) diverges
    like (R-r)^-1 and the integrand carries a (R-r)^-c1 term inside the exponential. A single
    Gauss-Legendre panel cannot resolve that endpoint, so the interval is split into panels
    whose widths halve towards the wall.
    """
    if dP_tip <= 0:
        return 0.0
    edges = [0.0] + [R * (1.0 - 0.5 ** k) for k in range(1, 41)] + [R]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        rv = 0.5 * (hi - lo) * (_GL_X + 1.0) + lo
        w = 0.5 * (hi - lo) * _GL_W
        tau_s = dP_tip * rv / (2 * LTIP)
        V_r = np.maximum(velocity_profile(dP_tip, R, rv), 1e-14)
        integ = rv * (1.0 - np.exp(-A1 * tau_s ** B1 * (LTIP / V_r) ** C1))
        integ = np.where(tau_s < 1e-12, 0.0, integ)
        total += float(np.dot(w, integ))
    return min(2.0 / R ** 2 * total, 1.0)


def compute_Dst_flux(dP_tip: float, R: float, Q_uLs: float) -> float:
    """Cell-flux-weighted shear damage, the population actually collected in the extrudate.

        D_st,Q = (2 pi / Q) * integral_0^R r * v_z(r) * D_s(r) dr

    Reported alongside `compute_Dst`, never in place of it: Ning inferred a1, b1 and c1 under
    the area-average convention, and validated the law against printed-cell damage using it,
    so Eq. (9) stays the primary quantity and this one describes collected cells.
    """
    if dP_tip <= 0 or Q_uLs <= 0:
        return 0.0
    Q = Q_uLs * 1e-9
    edges = [0.0] + [R * (1.0 - 0.5 ** k) for k in range(1, 41)] + [R]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        rv = 0.5 * (hi - lo) * (_GL_X + 1.0) + lo
        w = 0.5 * (hi - lo) * _GL_W
        tau_s = dP_tip * rv / (2 * LTIP)
        V_r = np.maximum(velocity_profile(dP_tip, R, rv), 1e-14)
        integ = rv * V_r * (1.0 - np.exp(-A1 * tau_s ** B1 * (LTIP / V_r) ** C1))
        integ = np.where(tau_s < 1e-12, 0.0, integ)
        total += float(np.dot(w, integ))
    return min(2.0 * math.pi / Q * total, 1.0)


def calibration_fraction(dP_tip: float, R: float, Q_uLs: float) -> float:
    """Flux-weighted fraction of the nozzle cross-section inside Ning's calibrated box.

    A cell at radius r sees (tau_s(r), t_s(r)). The pair is inside the box when the stress
    falls in 100-1000 Pa and the residence time in 10-120 s. Weighted by r*v_z(r), so the
    fraction counts cells delivered per unit time rather than area.
    """
    if dP_tip <= 0 or Q_uLs <= 0:
        return 0.0
    Q = Q_uLs * 1e-9
    inside = 0.0
    edges = [0.0] + [R * (1.0 - 0.5 ** k) for k in range(1, 41)] + [R]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        rv = 0.5 * (hi - lo) * (_GL_X + 1.0) + lo
        w = 0.5 * (hi - lo) * _GL_W
        tau_s = dP_tip * rv / (2 * LTIP)
        V_r = np.maximum(velocity_profile(dP_tip, R, rv), 1e-14)
        t_s = LTIP / V_r
        ok = ((tau_s >= SHEAR_BOX_TAU_PA[0]) & (tau_s <= SHEAR_BOX_TAU_PA[1])
              & (t_s >= SHEAR_BOX_TIME_S[0]) & (t_s <= SHEAR_BOX_TIME_S[1]))
        inside += float(np.dot(w, rv * V_r * ok))
    return min(2.0 * math.pi / Q * inside, 1.0)


def compute_Det(tau_ext: float) -> float:
    """Extensional damage, Ning Eq. (19). An exponential saturation, not a power law."""
    return 0.0 if tau_ext <= 0 else min(1.0 - math.exp(-A2 * tau_ext ** B2), 1.0)


def derive(P_kPa: float, Q_uLs: float, R: float) -> dict:
    """The full chain, from the two converged quantities (P_applied, Q) alone.

    tau_e is an *effective pressure-derived stress under the Cogswell-Ning closure*: the
    entrance residual it is formed from can also carry upstream wall friction, contraction
    shear, entrance development and kinetic-energy flux, so it is not a directly resolved
    extensional stress.
    """
    tw = tau_w_analytic(Q_uLs, R)
    dP_tip = 2 * LTIP * tw / R
    P_en = max(P_kPa * 1e3 - dP_tip, 0.0)
    tau_ext = 3 * (N_EXP + 1) / 8 * P_en
    Dst = compute_Dst(dP_tip, R)
    Dst_Q = compute_Dst_flux(dP_tip, R, Q_uLs)
    Det = compute_Det(tau_ext)
    Dt = (Det + (1 - Det) * Dst) * 100.0
    u_mean = (Q_uLs * 1e-9) / (math.pi * R ** 2)
    eta_w = K_VIS * (u_mean / R) ** (N_EXP - 1) if u_mean > 1e-12 else float("inf")
    Re = RHO * u_mean * 2 * R / eta_w if u_mean > 1e-12 else 0.0
    return dict(tau_w_Pa=tw, dP_tip_kPa=dP_tip / 1e3, P_en_kPa=P_en / 1e3,
                tau_ext_Pa=tau_ext, Dst_pct=Dst * 100, Det_pct=Det * 100,
                Dt_pct=Dt, Re=Re,
                Dst_flux_pct=Dst_Q * 100,
                shear_box_frac=calibration_fraction(dP_tip, R, Q_uLs),
                ext_in_box=int(EXT_BOX_TAU_PA[0] <= tau_ext <= EXT_BOX_TAU_PA[1]))


REFERENCE = dict(P_kPa=200.0, Q_uLs=17.940002925163476, R=100e-6,
                 tau_w_Pa=981.2178136160567, dP_tip_kPa=196.24356272321134,
                 P_en_kPa=3.756437276788667, tau_ext_Pa=1901.696371374263,
                 Dst_pct=0.28642409650113515, Det_pct=8.694829839414187,
                 Dt_pct=8.956349848105468)


def _self_test() -> int:
    """Reproduce the reference row. Non-zero exit if the chain has moved."""
    got = derive(REFERENCE["P_kPa"], REFERENCE["Q_uLs"], REFERENCE["R"])
    bad = []
    for k, want in REFERENCE.items():
        if k in ("P_kPa", "Q_uLs", "R"):
            continue
        rel = abs(got[k] - want) / abs(want)
        flag = "ok " if rel < 1e-12 else "BAD"
        if rel >= 1e-12:
            bad.append(k)
        print(f"  {flag} {k:14s} {got[k]:>20.10f}   expected {want:>20.10f}   rel {rel:.2e}")
    print(f"\n  Dst area {got['Dst_pct']:.4f} %   flux {got['Dst_flux_pct']:.4f} %"
          f"   flux-weighted fraction inside Ning's shear box {100*got['shear_box_frac']:.1f} %"
          f"   tau_e in box: {'yes' if got['ext_in_box'] else 'no'}")
    if bad:
        print("\n  FAILED: " + ", ".join(bad))
        return 1
    print("\n  PASS - the damage chain reproduces the reference row.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
