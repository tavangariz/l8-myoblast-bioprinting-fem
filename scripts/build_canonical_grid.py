#!/usr/bin/env python3
"""Rebuild the 32-case grid (four nozzle diameters, eight applied pressures) from the solver output.

Reads the per-case solver files in results/fem/runs/2026-08-26_ning-grid32 (meshes M12 and M16)
and the Bagley series in results/fem/runs/2026-08-27_bagley-ning, reduces every case through
l8_damage.derive, and writes results/fem/exports/L8_FEM_canonical_2026-08-27.csv, the grid the
results of the article are read from.

Nothing is written unless three checks pass: the damage law reproduces its reference row, the
entrance pressure agrees with a Bagley extrapolation to zero nozzle length within 3 % for every
nozzle, and it changes by less than 1 % between meshes M12 and M16.

    python scripts/build_canonical_grid.py
"""
from __future__ import annotations

import csv
import hashlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from l8_damage import derive, _self_test          # noqa: E402
from l8_geometry import TIP_RADIUS_M              # noqa: E402
from l8_paths import DATA, require                # noqa: E402

RUNS = DATA / "fem" / "runs"
EXPORTS = DATA / "fem" / "exports"
OUT = EXPORTS / "L8_FEM_canonical_2026-08-27.csv"

GRID = RUNS / "2026-08-26_ning-grid32"
BAGLEY = RUNS / "2026-08-27_bagley-ning"

DIAS = [100, 200, 250, 330]
SOURCE_M12 = {d: GRID / f"ning_grid_M12_D{d}.csv" for d in DIAS}
SOURCE_M16 = {d: GRID / f"ning_grid_M16_D{d}.csv" for d in DIAS}
BAGLEY_L = {"2p5mm": 2.5, "5mm": 5.0, "10mm": 10.0}

BAGLEY_TOL_PCT = 3.0
MESH_TOL_PCT = 1.0


def load_column(path: pathlib.Path) -> dict:
    require(path, f"solver output {path.name}")
    return {float(r["pressure_kPa"]): r for r in csv.DictReader(open(path))}


def bagley_intercept(d: int, target_Q: float):
    """Applied pressure extrapolated to zero tip length at a fixed flow rate.

    Returns (intercept kPa, standard error kPa, whether target_Q needed extrapolating).
    The standard error matters: three lengths and two parameters leaves one degree of
    freedom, and an intercept quoted without it invites a comparison the data cannot carry.
    """
    Ls, Ps, extrapolated = [], [], False
    for tag, L in sorted(BAGLEY_L.items(), key=lambda kv: kv[1]):
        f = BAGLEY / f"bagley_ning_D{d}_L{tag}.csv"
        require(f, f"Bagley series {f.name}")
        rows = list(csv.DictReader(open(f)))
        q = np.array([float(r["Q_uLs"]) for r in rows])
        p = np.array([float(r["pressure_kPa"]) for r in rows])
        if not (q.min() <= target_Q <= q.max()):
            extrapolated = True
        cf = np.polyfit(np.log(q), np.log(p), 1)          # log-log, see the module docstring
        Ls.append(L)
        Ps.append(float(np.exp(np.polyval(cf, np.log(target_Q)))))
    coef, cov = np.polyfit(np.array(Ls), np.array(Ps), 1, cov=True)
    return float(coef[1]), float(np.sqrt(cov[1, 1])), extrapolated


def main() -> int:
    print("Canonical grid — 4 tips x 8 pressures, analytic wall shear stress\n")

    print("  GATE 1 — the damage chain reproduces its reference row\n")
    if _self_test() != 0:
        sys.exit("REFUSING: the damage chain no longer reproduces its reference row")

    print("\n  GATE 2 — Bagley cross-check (analytic P_en vs zero-length extrapolation)\n")
    gate_pct, bagley_kPa, hard_fail = {}, {}, []
    for d in DIAS:
        # The Bagley runs are at M16, so the comparison is made at M16: the target flow rate and the
        # analytic value both come from the M16 column.
        Q200 = float(load_column(SOURCE_M16[d])[200.0]["Q_uLs"])
        pen_an = derive(200.0, Q200, TIP_RADIUS_M[d])["P_en_kPa"]
        pen_bag, se, extrap = bagley_intercept(d, Q200)
        err = 100 * (pen_an / pen_bag - 1)
        gate_pct[d], bagley_kPa[d] = err, pen_bag
        flag = "" if abs(err) <= BAGLEY_TOL_PCT else "   <-- OVER TOLERANCE"
        note = "  (target Q extrapolated at one length)" if extrap else ""
        print(f"    Ø{d:<4d} analytic {pen_an:8.3f}  Bagley {pen_bag:8.3f} ± {se:5.3f} kPa"
              f"  -> {err:+7.2f} %   amplification {200.0/pen_an:5.1f}x{flag}{note}")
        if abs(err) > BAGLEY_TOL_PCT:
            hard_fail.append(f"Ø{d} disagrees with Bagley by {err:+.1f} %")
    if hard_fail:
        sys.exit("REFUSING: " + "; ".join(hard_fail) +
                 f" — tolerance is {BAGLEY_TOL_PCT} % and is not adjustable here")
    print(f"\n    PASS — worst disagreement {max(abs(v) for v in gate_pct.values()):.2f} %,"
          f" tolerance {BAGLEY_TOL_PCT} %")

    print("\n  GATE 3 — mesh stability of the analytic route (M12 -> M16)\n")
    worst_mesh = 0.0
    for d in DIAS:
        a = derive(200.0, float(load_column(SOURCE_M12[d])[200.0]["Q_uLs"]), TIP_RADIUS_M[d])
        b = derive(200.0, float(load_column(SOURCE_M16[d])[200.0]["Q_uLs"]), TIP_RADIUS_M[d])
        drift = 100 * (b["P_en_kPa"] / a["P_en_kPa"] - 1)
        worst_mesh = max(worst_mesh, abs(drift))
        print(f"    Ø{d:<4d} P_en M12->M16 drift {drift:+.3f} %")
    if worst_mesh > MESH_TOL_PCT:
        sys.exit(f"REFUSING: analytic P_en drifts {worst_mesh:.2f} % with mesh")
    print(f"\n    PASS — worst drift {worst_mesh:.3f} %, tolerance {MESH_TOL_PCT} %")

    rows = []
    for d in DIAS:
        col = load_column(SOURCE_M12[d])
        for P in sorted(col):
            out = derive(P, float(col[P]["Q_uLs"]), TIP_RADIUS_M[d])
            # The published columns, in the published order.
            out = {k: v for k, v in out.items()
                   if k not in ("Dst_flux_pct", "shear_box_frac", "ext_in_box")}
            rows.append(dict(needle_um=d, pressure_kPa=P, mesh_M=12, inertia=1,
                             Q_uLs=float(col[P]["Q_uLs"]), **out,
                             pen_bagley_kPa=bagley_kPa[d], pen_gate_pct=gate_pct[d],
                             source=SOURCE_M12[d].name))

    rows.sort(key=lambda r: (r["needle_um"], r["pressure_kPa"]))
    press = sorted({r["pressure_kPa"] for r in rows})
    if len(rows) != len(DIAS) * len(press):
        sys.exit(f"REFUSING: {len(rows)} rows is not {len(DIAS)} x {len(press)}")
    seen = set()
    for r in rows:
        key = (r["needle_um"], r["pressure_kPa"])
        if key in seen:
            sys.exit(f"REFUSING: duplicate case {key}")
        seen.add(key)

    cols = ["needle_um", "pressure_kPa", "mesh_M", "inertia", "Q_uLs", "Re", "tau_w_Pa",
            "dP_tip_kPa", "P_en_kPa", "tau_ext_Pa", "Dst_pct", "Det_pct", "Dt_pct",
            "pen_bagley_kPa", "pen_gate_pct", "source"]
    EXPORTS.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(f"{digest}  {OUT.name}\n")
    print(f"\n  wrote {len(rows)} rows -> {OUT.relative_to(DATA.parent)}")
    print(f"  sha256 {digest}")

    by = {(r["needle_um"], r["pressure_kPa"]): r for r in rows}
    print("\n  CANONICAL D_t (%)\n")
    print("    kPa  " + "".join(f"{d:>10d}" for d in DIAS))
    for P in press:
        print(f"    {P:<5.0f}" + "".join(f"{by[(d, P)]['Dt_pct']:10.3f}" for d in DIAS))
    h = by[(330, 200.0)]["Dt_pct"] / by[(200, 200.0)]["Dt_pct"]
    print(f"\n  Ø200 -> Ø330 at 200 kPa: {by[(200, 200.0)]['Dt_pct']:.3f} -> "
          f"{by[(330, 200.0)]['Dt_pct']:.3f} % = {100 * (h - 1):+.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
