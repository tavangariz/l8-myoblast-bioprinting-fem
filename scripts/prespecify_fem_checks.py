"""Fix the eleven FEM checks of the surrogate before any of them is solved.

The cases are chosen from the surrogate's predictions, all inside the trained domain
(25-400 kPa, 100-330 µm): the operating point of each diameter (lowest predicted Dt with
Dt < 20 % and Q >= 1 µL/s), the pressure at which the predicted Dt reaches 20 % where it exists,
and one interior condition. The diameters 150 and 290 µm are not in the grid. Writes the case
list with the predictions (PRESPECIFIED_CASES.csv) and the HTCondor submit file to
results/fem/runs/2026-09-23_independent-checks/.

    python scripts/prespecify_fem_checks.py
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

import l8_surrogate as S

RUN = "2026-09-23_independent-checks"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "fem" / "runs" / RUN
TRAINED_D = (100.0, 200.0, 250.0, 330.0)
INTERMEDIATE_D = (150.0, 290.0)
INTERIOR = ((150.0, 300.0),)            # (diameter µm, pressure kPa)
MESH_M = 12                             # the canonical grid's mesh


def dt_boundary(gp_dt, z, d, n=3751):
    """Lowest pressure in the trained range at which predicted D_t reaches DT_MAX."""
    P = np.linspace(*S.P_TRAINED, n)
    dt = gp_dt.predict(z(P, np.full_like(P, d)))
    hit = np.flatnonzero(dt >= S.DT_MAX)
    return None if hit.size == 0 else float(P[hit[0]])


def main():
    gp_dt, gp_q, z = S.fit_all(check=True)
    cases = []
    for d in TRAINED_D + INTERMEDIATE_D:
        op = S.operating_point(gp_dt, gp_q, z, d)
        if op is None:
            print(f"  Ø{d:.0f}: no feasible condition")
        else:
            cases.append((d, round(op["P_kPa"], 1), "optimum"))
        pb = dt_boundary(gp_dt, z, d)
        if pb is not None:
            cases.append((d, round(pb, 1), "dt_boundary"))
    cases += [(d, p, "interior") for d, p in INTERIOR]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, p, role in cases:
        Xz = z([p], [d])
        m_dt, s_dt = (float(v[0]) for v in gp_dt.predict(Xz, return_std=True))
        m_q, s_q = (float(v[0]) for v in gp_q.predict(Xz, return_std=True))
        rows.append({"case": f"M{MESH_M}_D{d:g}_P{p:g}", "needle_um": d, "pressure_kPa": p,
                     "role": role, "gp_Dt_pct": round(m_dt, 4), "gp_Dt_sd_pp": round(s_dt, 4),
                     "gp_log10_Q": round(m_q, 5), "gp_log10_Q_sd": round(s_q, 5),
                     "gp_Q_uLs": round(10 ** m_q, 5)})
    with open(OUT / "PRESPECIFIED_CASES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    queue = "\n".join(f"  {MESH_M}, {r['needle_um']:g}, {r['pressure_kPa']:g}" for r in rows)
    (OUT / "l8_independent_checks.sub").write_text(SUB.format(run=RUN, queue=queue))
    for r in rows:
        print(f"  {r['case']:<20} {r['role']:<12} D_t {r['gp_Dt_pct']:6.2f} ± {r['gp_Dt_sd_pp']:.2f} pp"
              f"   Q {r['gp_Q_uLs']:9.3f} µL/s")
    print(f"  {len(rows)} cases -> {OUT.relative_to(OUT.parents[3])}")


SUB = """\
# Independent FEM checks of the GP surrogate, fixed by prespecify_fem_checks.py
# before any case was solved. Same solver, image and settings as the canonical grid;
# only the run directory and the case list differ.
universe                = vanilla
getenv                  = True
should_transfer_files   = YES
transfer_executable     = False
when_to_transfer_output = ON_EXIT

executable = /usr/bin/apptainer

request_cpus   = 1
request_memory = 8 GB
request_disk   = 10 GB

arguments = "exec $ENV(HOME)/htcondor/images/l8-fem.sif python3 -u $ENV(HOME)/l8-fem/scripts/fem_bagley_inertia_2026-07-30.py"

RUNDIR = $ENV(HOME)/l8-fem/results/fem/runs/{run}
TAG    = M$(MESH)_D$(DIA)_P$(PRESS)

environment = "L8_GEOM=ning L8_EPS_AX=1e-12 L8_MESH_M=$(MESH) L8_INERTIA=1 L8_NEEDLES=$(DIA) L8_PRESSURES=$(PRESS) L8_CSV_OUT=$(RUNDIR)/$(TAG).csv MPLBACKEND=Agg OMP_NUM_THREADS=1"

output = $(RUNDIR)/logs/$(TAG).out
error  = $(RUNDIR)/logs/$(TAG).err
log    = $(RUNDIR)/logs/$(TAG).log

queue MESH, DIA, PRESS from (
{queue}
)
"""

if __name__ == "__main__":
    main()
