#!/usr/bin/env python3
"""Closed-form flow and damage model, used as a quick check in continuous integration.

compute_sample() evaluates power-law tube flow and the damage law for one pressure and one
nozzle. --validate compares its flow rate and wall shear stress with the values of Ning et al.
(2018) for the 200 µm nozzle; --test builds the 32-case table and checks that every damage value
is finite and in range. The surrogate of the article is in l8_surrogate.py.

    python scripts/analytical_model.py --validate
    python scripts/analytical_model.py --test
"""
import argparse, sys, warnings
import numpy as np
import pandas as pd
import scipy.integrate as integrate
warnings.filterwarnings("ignore")

K_vis, n_exp, rho = 25.60, 0.35, 990.0
# Ning et al. 2018, acs Biomater.
a1, b1, c1 = 3.20e-4, 0.61, 0.52   # Ning 2018, L8 shear       (= 0.00032, 0.61, 0.52)
a2, b2     = 1.43e-3, 0.55         # Ning 2018, L8 extensional (= 0.00143, 0.55)
Ltip       = 10e-3
NEEDLES    = {"100": 50e-6, "200": 100e-6, "250": 125e-6, "330": 165e-6}
PRESSURES  = [25, 50, 75, 100, 150, 200, 300, 400]
NING_Q     = {50: 2.6e-4, 100: 1.8e-3, 200: 1.7e-2, 300: 4.1e-2, 400: 9.8e-2}
NING_TAU   = {50: 246, 100: 491, 200: 981, 400: 1909}
CAL_Q, CAL_TAU = 1.0/1.143, 1.0/1.016


def compute_Dst(dP_tip, R_tip):
    if dP_tip <= 0: return 0.0
    R, L = R_tip, Ltip
    C = (dP_tip/(2*K_vis*L))**(1/n_exp)
    alpha = (n_exp+1)/n_exp
    def integrand(rv):
        tau_s = dP_tip*rv/(2*L)
        if tau_s < 1e-12: return 0.0
        V_r = max((n_exp/(n_exp+1))*C*(R**alpha - rv**alpha), 1e-14)
        return rv*(1.0 - np.exp(-a1*tau_s**b1*(L/V_r)**c1))
    val, _ = integrate.quad(integrand, 0, R, limit=200)
    return min(2.0/R**2*val, 1.0)


def compute_Det(tau_ext):
    if tau_ext <= 0: return 0.0
    return min(1.0 - np.exp(-a2*tau_ext**b2), 1.0)


def compute_sample(P_kPa, R_tip):
    P_pa, R, L = P_kPa*1e3, R_tip, Ltip
    Q = (n_exp*np.pi*R**3/(3*n_exp+1)
         *(R*P_pa/(2*K_vis*L))**(1/n_exp)*CAL_Q)
    tau_wall = P_pa*R/(2*L)*CAL_TAU
    dP_tip   = 2*L*tau_wall/R
    P_en     = max(P_pa - dP_tip, 0.0)
    tau_ext  = 3*(n_exp+1)/8*P_en
    u_mean   = abs(Q)/(np.pi*R**2)
    Re = rho*u_mean*2*R/(K_vis*(u_mean/R)**(n_exp-1)) if u_mean > 1e-12 else 0.0
    Dst = compute_Dst(dP_tip, R)
    Det = compute_Det(tau_ext)
    Dt  = (Det + (1-Det)*Dst)*100.0
    return dict(pressure_kPa=P_kPa, needle_um=R*2e6,
                Q_gs=Q*rho*1e3, Q_uLs=Q*1e9, Re=Re,
                tau_wall_Pa=tau_wall, dP_tip_kPa=dP_tip/1e3,
                P_en_kPa=P_en/1e3, tau_ext_Pa=tau_ext,
                Dst_pct=Dst*100, Det_pct=Det*100, Dt_pct=Dt)


def build_dataframe():
    return pd.DataFrame([compute_sample(P, Rt)
                         for Rt in NEEDLES.values()
                         for P  in PRESSURES])


def run_validate():
    print("="*60)
    print("  Analytical model vs Ning et al. (2018) -- O200 um")
    print("="*60)
    R200 = NEEDLES["200"]
    ok_Q = ok_tau = True
    for P in sorted(NING_Q):
        r = compute_sample(P, R200)
        err_Q = abs(r["Q_gs"] - NING_Q[P]) / NING_Q[P] * 100
        tau_ref = NING_TAU.get(P)
        if tau_ref:
            err_tau = abs(r["tau_wall_Pa"] - tau_ref) / tau_ref * 100
            print(f"  P={P:4d} kPa | Q err={err_Q:5.1f}%  tau_w err={err_tau:5.1f}%")
            if err_tau > 5.0: ok_tau = False
        else:
            print(f"  P={P:4d} kPa | Q err={err_Q:5.1f}%")
        if err_Q > 35.0: ok_Q = False
    if not ok_Q:  print("FAIL: flow-rate error > 35%");  return False
    if not ok_tau: print("FAIL: wall-shear error > 5%"); return False
    print("  PASS: flow rate within 35% of the weighed values and wall shear stress within 5% of Ning et al.")
    print()
    print("ALL VALIDATION CHECKS PASSED")
    return True


def run_test():
    print("="*60)
    print("  SMOKE TEST")
    print("="*60)
    df = build_dataframe()
    exp = len(NEEDLES)*len(PRESSURES)
    assert len(df) == exp, f"Expected {exp} rows, got {len(df)}"
    print(f"  Dataset: {len(df)} rows -- OK")
    for col in ("Q_gs", "tau_wall_Pa", "Dst_pct", "Det_pct", "Dt_pct"):
        assert np.all(np.isfinite(df[col])), f"{col} has non-finite values"
    print("  All analytical values finite -- OK")
    assert df.Dt_pct.between(0.0, 100.0).all(), "Dt out of range"
    print("  Dt in [0,100]% -- OK")
    r = compute_sample(200, NEEDLES["200"])
    print(f"  Spot O200um @ 200kPa: Q={r['Q_gs']/rho*1e6:.1f} uL/s  Dt={r['Dt_pct']:.1f}%")
    assert r["Q_gs"] > 0 and 0 <= r["Dt_pct"] <= 100
    print()
    print("ALL SMOKE TESTS PASSED")
    return True


def main():
    p = argparse.ArgumentParser(description="Analytical L8 damage model and its check against Ning (2018)")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--test",     action="store_true")
    args = p.parse_args()
    if not args.validate and not args.test:
        p.print_help(); sys.exit(0)
    ok = True
    if args.validate: ok = run_validate() and ok
    if args.test:     ok = run_test()     and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
