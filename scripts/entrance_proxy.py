#!/usr/bin/env python3
"""The entrance stress derived from the pressure, against the resolved flow at the contraction.

The damage law turns the entrance residual p_en into an effective stress tau_e = 3(n+1)/8 p_en (Cogswell-Ning
closure). This script asks whether that proxy follows the extension the flow imposes, from solved fields at the
reference condition (200 µm, 200 kPa, Re 1.2) and the most inertial case of the grid (330 µm, 400 kPa, Re 195),
on meshes M12 and M16, and at two points of the 20 % limit (results/fem/runs/2026-10-31_boundary-fields). For
each field, along the axis through the cartridge-to-nozzle step:

  peak extensional strain rate  max du_z/dz
  Hencky strain                 ln(u_z,out / u_z,in), the total stretch a cell on the axis receives
  exposure time                 time spent where du_z/dz exceeds half its peak
  resolved axial stress         3 eta(sqrt(3) eps_dot) eps_dot at the peak, generalized-Newtonian,
                                with the solver's bounded power-law viscosity

and the ratio of each between the first two fields, next to the ratio of tau_e (Additional file 1, Section
S13, Table S16). The solved fields (.pkl, written by the solver with L8_VIZ_CKPT) are not distributed except
the reference one; writes results/analyses/entrance_proxy.json.

    python scripts/entrance_proxy.py REF_PKL HIGH_PKL [MORE_PKL ...]
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from l8_damage import K_VIS, N_EXP, derive
from l8_paths import DATA

ETA_MIN, ETA_MAX = 0.003, 500.0          # the solver's viscosity bounds (Methods)
NZ = 6000


def axis_metrics(pkl: str) -> dict:
    d = pickle.load(open(pkl, "rb"))
    prov = d["provenance"]
    c = d["vtk_coords"]
    r, z = c[:, 0], c[:, 1]
    uz = LinearNDInterpolator(np.column_stack([z, r]), d["fields"]["uvec"][:, 1])
    z2 = d["z2"]
    zg = np.linspace(z2 - 2.0e-3, z2 + 1.5e-3, NZ)
    u = uz(zg, np.zeros_like(zg))
    ok = np.isfinite(u)
    zg, u = zg[ok], u[ok]
    eps = np.gradient(u, zg)
    ipk = int(np.argmax(eps))
    pk = float(eps[ipk])
    half = eps >= 0.5 * pk
    # exposure: integrate dz/u over the contiguous half-peak region around the peak
    lo = ipk
    while lo > 0 and half[lo - 1]:
        lo -= 1
    hi = ipk
    while hi < len(zg) - 1 and half[hi + 1]:
        hi += 1
    t_exp = float(np.trapz(1.0 / u[lo:hi + 1], zg[lo:hi + 1]))
    u_in, u_out = float(u[0]), float(np.max(u))
    gd = np.sqrt(3.0) * pk
    eta = float(np.clip(K_VIS * gd ** (N_EXP - 1), ETA_MIN, ETA_MAX))
    D_um, P = float(prov["needle_um"]), float(prov["pressure_kPa"])
    ch = derive(P, float(d["Q_uLs"]), d["Rtip"])
    return dict(file=pkl.split("/")[-1], D_um=D_um, P_kPa=P, Re=round(ch["Re"], 2),
                peak_strain_rate_s=round(pk, 1), fwhm_um=round(float((zg[hi] - zg[lo]) * 1e6), 1),
                hencky_strain=round(float(np.log(u_out / u_in)), 3), exposure_ms=round(1e3 * t_exp, 4),
                resolved_axial_stress_Pa=round(3 * eta * pk, 1), tau_e_Pa=round(ch["tau_ext_Pa"], 1),
                P_en_kPa=round(ch["P_en_kPa"], 3), Det_pct=round(ch["Det_pct"], 3))


def main():
    files = sys.argv[1:]
    res = [axis_metrics(f) for f in files]
    lo, hi = res[0], res[1]
    ratio = {k: round(hi[k] / lo[k], 3) for k in ("peak_strain_rate_s", "hencky_strain", "exposure_ms",
                                                    "resolved_axial_stress_Pa", "tau_e_Pa")}
    out = dict(cases=res, ratio_high_over_reference=ratio)
    (DATA / "analyses").mkdir(parents=True, exist_ok=True)
    (DATA / "analyses" / "entrance_proxy.json").write_text(json.dumps(out, indent=1))
    for c in res:
        print(c)
    print("ratio high/reference:", ratio)


if __name__ == "__main__":
    main()
