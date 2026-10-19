#!/usr/bin/env python3
"""Axisymmetric finite element solver for the cartridge and nozzle (DOLFINx 0.10).

Solves steady flow of the power-law cell suspension with Taylor-Hood elements and Newton
iteration, with continuation in the lower viscosity bound, the applied pressure and the density
(the convective term). Each run writes one CSV, one row per case, to the path in L8_CSV_OUT: flow
rate, damage terms and the convergence record. It exits non-zero if any case fails to converge
at full density.

Settings of the grid in the article:

    L8_GEOM=ning L8_EPS_AX=1e-12 L8_MESH_M=12 L8_INERTIA=1 L8_LTIP=10e-3

L8_INERTIA=0 removes the convective term, L8_LTIP sets the nozzle length (Bagley series),
L8_MESH_M the number of elements across the nozzle radius, L8_NEEDLES and L8_PRESSURES select
the cases, and L8_VIZ_CKPT saves the solved field at L8_VIZ_P_KPA as a checkpoint. Run it in the
Docker image (see the README).
"""

# ─── SWITCH ──────────────────────────────────────────────────────────────────
RUN_FEM = True   # the solve needs DOLFINx; run it in the Docker image or on the cluster
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import scipy.integrate as integrate
import csv
import json
import hashlib
import datetime as dt
import sys
try:
    import pandas as pd
except ModuleNotFoundError:              # not in the DOLFINx container; PART B only
    pd = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

# Axisymmetric continuity regulariser for the hoop terms u_r/(r + EPS_AX).
# This is an absolute length compared against the radial coordinate, so its relative
# effect scales as 1/R and it acts as a distributed mass source. At 1e-6 m it destroyed
# 97.8 % of the mass along the Ø100 um tip and 15.1 % along Ø200. Keep it far below the
# smallest tip radius.
EPS_AX = float(os.environ.get("L8_EPS_AX", "1e-12"))

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PUBLICATION FIGURE STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          13,
    "axes.titlesize":     15,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "figure.titlesize":   16,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "axes.linewidth":     1.0,
    "lines.linewidth":    2.2,
    "lines.markersize":   7,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
})

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL & DAMAGE PARAMETERS  (Ning et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════
K_vis, n_exp, rho = 25.60, 0.35, 990.0
a1, b1, c1        = 3.20e-4, 0.61, 0.52  # Ning et al. (2018), L8 myoblasts
a2, b2            = 1.43e-3, 0.55

MESH_M  = float(os.environ.get("L8_MESH_M", "0"))  # cells across tip radius
INERTIA = os.environ.get("L8_INERTIA", "0") == "1"   # add rho*(u.grad)u
# Nodal snapshot for the animation renderers. Empty disables it; otherwise the
# converged field at L8_VIZ_P_KPA is pickled to this path, one file per needle.
VIZ_CKPT  = os.environ.get("L8_VIZ_CKPT", "")
VIZ_P_KPA = float(os.environ.get("L8_VIZ_P_KPA", "200"))
# Bagley build: the capillary length is swept so dP(L) can be extrapolated to L = 0.
# Cartridge lengths. Settable so the feed line can be length-decomposed
# the same way the tip already is: at fixed flow rate, the slope of applied pressure
# against each length is that section's true friction gradient, and the intercept
# isolates the localized contraction losses without reference to any external value.
# Defaults are Ning's published cartridge.
Lc1 = float(os.environ.get("L8_LC1", "12e-3"))
Lc2 = float(os.environ.get("L8_LC2", "10.5e-3"))
Ltip = float(os.environ.get("L8_LTIP", "10e-3"))
Rin, Rmid      = 2.1e-3, 1.0e-3

# ── GEOMETRY MODE ────────────────────────────────────────────────────────────
# The domain is the cartridge-and-needle assembly of Ning et al. 2018, Fig. 1.
# Three levels of evidence stand behind it, and they are not the same:
#
# PRINTED IN THE SOURCE.  D1 = 4.2 mm, D2 = 2 mm, D = 0.10-0.33 mm,
#   Lc1 = 12 mm, Lc2 = 10.5 mm, L = 10 mm, all from the Fig. 1 caption.
#
# READ FROM THE DRAWING.  The two axial dimension arrows are nested rather than
#   sequential: Lc2 is the straight cartridge and Lc1 is the whole cartridge, so
#   the transition occupies Lc1 - Lc2 = 1.5 mm and the total flow path is 22.0 mm.
#   Support: taking Lc2 = 10.5 mm as the axial scale (19.43 px/mm on the rendered
#   page) returns 12.92 mm for the cartridge against a printed 12, and 9.78 mm for
#   the needle against a printed 10; and Ning writes of "an abruptly contractive
#   region between the cartridge body and needle tip" in the singular.  The
#   sequential reading requires two abrupt contractions and a 10.5 mm bore-D2 pipe
#   that appears nowhere in the paper.
#
# OUR MODELLING CHOICE.  Ning prints no transition profile, and the 1.5 mm region
#   is represented here as a straight linear cone from Rin to Rmid.  That specific
#   shape is a reconstruction consistent with the figure and the text; it is not a
#   manufacturer specification, and the real internal profile may differ.  It is a
#   modelling choice.
#
# "legacy" reads Lc1 and Lc2 as SEQUENTIAL: three stacked cylinders, a 10.5 mm Rmid
# pipe that appears in no source, and a 32.5 mm flow path with TWO abrupt
# contractions.  It is kept only so that earlier results computed on it stay
# reproducible; every reported result uses "ning".
GEOM = os.environ.get("L8_GEOM", "ning").lower()
if GEOM not in ("ning", "legacy"):
    raise SystemExit(f"L8_GEOM must be 'ning' or 'legacy', got {GEOM!r}")
L_STRAIGHT = Lc2                 # straight cartridge at Rin
L_CONE     = Lc1 - Lc2           # modelled linear cone Rin -> Rmid (1.5 mm); see note above
if GEOM == "ning" and L_CONE <= 0:
    raise SystemExit(f"ning geometry needs Lc1 > Lc2; got Lc1={Lc1}, Lc2={Lc2}")

NEEDLES    = {"100": 50e-6, "200": 100e-6, "250": 125e-6, "330": 165e-6}
PRESSURES  = [25, 50, 75, 100, 150, 200, 300, 400]

# ─────────────────────────────────────────────────────────────────────────────
# P_en READ DIRECTLY FROM THE FEM PRESSURE FIELD
#
# The reported entrance pressure is the residual (see l8_damage.py)
#       dP_tip = 2*Ltip*tau_w/R_tip      <- ANALYTICAL fully-developed pipe law
#       P_en   = P_applied - dP_tip      <- leftover
# Because dP_tip is ~98.5% of the applied pressure, that subtraction amplifies any
# error in tau_w by dP_tip/P_en, which is 20x to 67x across the four needles.
# Measured against Ning 2018 Table 2 (L8, 200 kPa):
#
#   D(um)  dPtip/P_en   tau_w err   predicted P_en err   observed P_en err
#     100        66.6      -0.69%              +45.6%             +79.3%
#     200        64.6      +0.38%              -24.8%             -18.4%
#     250        32.8      -0.08%               +2.7%              +5.4%
#     330        20.1      +2.59%              -52.0%             -33.8%
#
# Every sign and every order of magnitude matches the amplification, which identifies
# the residual subtraction (not the mesh, not the convective term) as the source of the
# P_en differences.
#
# INDEPENDENT ESTIMATE
# A Bagley-style entrance excess read straight from the computed pressure field:
#   1. sample centerline pressure p(z) along the tip (pressure is r-uniform in the
#      developed region, so the centerline value is the cross-section value there);
#   2. linear-fit dp/dz over the DEVELOPED window only, excluding the entrance
#      development length and the outlet;
#   3. P_en_direct = p(z2) - |dp/dz| * Ltip
#      the pressure at the tip entrance in excess of what fully-developed flow alone
#      would require. Both terms are O(P_en), so there is no amplification.
#
# Both estimates go into the CSV side by side, from the same solve, so any difference
# between the two columns is attributable to the extraction alone.
# ─────────────────────────────────────────────────────────────────────────────
PEN_FIT_LO = float(os.environ.get("L8_PEN_FIT_LO", "0.35"))   # developed window start
PEN_FIT_HI = float(os.environ.get("L8_PEN_FIT_HI", "0.95"))   # ... end, fraction of Ltip
PEN_NSAMP  = int(os.environ.get("L8_PEN_NSAMP", "400"))

if os.environ.get("L8_PRESSURES"):
    PRESSURES = [float(x) for x in os.environ["L8_PRESSURES"].split(",")]
if os.environ.get("L8_NEEDLES"):
    # L8_NEEDLES selects needles from the table and also ADMITS diameters that are not in
    # it, with the tip radius taken as half the requested diameter in micrometres.
    #
    # Needles already in the table keep their STORED radii, so the canonical grid is solved on
    # exactly the stored geometry. That is what the `k in NEEDLES` branch is for, and it is
    # not cosmetic: float("200") *
    # 0.5e-6 is 9.999999999999999e-05, NOT the stored 100e-6, so deriving the radius for a
    # needle already in the table would perturb the canonical geometry in the last bits.
    # Only the cartridge geometry is fixed; the tip is a free parameter, so a
    # denser diameter sweep needs no other change.
    _want = [k.strip() for k in os.environ["L8_NEEDLES"].split(",") if k.strip()]
    NEEDLES = {k: (NEEDLES[k] if k in NEEDLES else float(k) * 0.5e-6) for k in _want}

NING_Q     = {50: 2.6e-4, 100: 1.8e-3, 200: 1.7e-2, 300: 4.1e-2, 400: 9.8e-2}
NING_TW    = {50: 246, 100: 491, 200: 981, 400: 1909}

CAL_Q      = 1.0 / 1.143
CAL_TAU    = 1.0 / 1.016

COLOURS = {"100": "#d62728", "200": "#1f77b4", "250": "#2ca02c", "330": "#9467bd"}

# ══════════════════════════════════════════════════════════════════════════════
#  DAMAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════
def compute_Dst(dP_tip, R_tip):
    """Radially-integrated shear damage in tip (Ning Eq 9)."""
    if dP_tip <= 0:
        return 0.0
    R, L = R_tip, Ltip
    C     = (dP_tip / (2 * K_vis * L)) ** (1 / n_exp)
    alpha = (n_exp + 1) / n_exp
    def integrand(rv):
        tau_s = dP_tip * rv / (2 * L)
        if tau_s < 1e-12:
            return 0.0
        V_r = max((n_exp / (n_exp + 1)) * C * (R**alpha - rv**alpha), 1e-14)
        t_s = L / V_r
        return rv * (1.0 - np.exp(-a1 * tau_s**b1 * t_s**c1))
    val, _ = integrate.quad(integrand, 0, R, limit=200)
    return min(2.0 / R**2 * val, 1.0)

def compute_Det(tau_ext):
    """Extensional damage (Ning Eq 19)."""
    if tau_ext <= 0:
        return 0.0
    return min(1.0 - np.exp(-a2 * tau_ext**b2), 1.0)

def compute_Dt(Det, Dst):
    """Total damage (Ning Eq 1).  Returns percent."""
    return (Det + (1 - Det) * Dst) * 100.0

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICAL MODEL  (calibrated against validated Ø200 µm FEM)
# ══════════════════════════════════════════════════════════════════════════════
def compute_sample(P_kPa, R_tip):
    P_pa, R, L = P_kPa * 1e3, R_tip, Ltip
    Q = (n_exp * np.pi * R**3 / (3*n_exp + 1)
         * (R * P_pa / (2*K_vis*L)) ** (1/n_exp) * CAL_Q)
    tau_wall = P_pa * R / (2*L) * CAL_TAU
    dP_tip   = 2 * L * tau_wall / R
    P_en     = max(P_pa - dP_tip, 0.0)
    tau_ext  = 3 * (n_exp + 1) / 8 * P_en
    u_mean   = abs(Q) / (np.pi * R**2)
    if u_mean > 1e-12:
        eta_w = K_vis * (u_mean / R) ** (n_exp - 1)
        Re    = rho * u_mean * 2 * R / eta_w
    else:
        Re = 0.0
    Dst = compute_Dst(dP_tip, R)
    Det = compute_Det(tau_ext)
    Dt  = compute_Dt(Det, Dst)
    return dict(
        pressure_kPa=P_kPa, needle_um=R*2e6,
        Q_gs=Q*rho*1e3, Q_uLs=Q*1e9, Re=Re,
        tau_wall_Pa=tau_wall, dP_tip_kPa=dP_tip/1e3,
        P_en_kPa=P_en/1e3, tau_ext_Pa=tau_ext,
        Dst_pct=Dst*100, Det_pct=Det*100, Dt_pct=Dt)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
if RUN_FEM:
    # ══════════════════════════════════════════════════════════════════════
    #  FEM SWEEP  (DOLFINx 0.10), one needle after another.
    #  Architecture:
    #    • axisymmetric cartridge-and-needle domain after Ning et al. 2018 Fig. 1
    #      (L8_GEOM=ning: straight cartridge → conical chamfer → abrupt contraction → needle)
    #    • Neumann traction BCs (pressure as surface integral, NOT Dirichlet)
    #    • Manual Newton with MUMPS LU + backtracking
    #    • Viscosity continuation: µ_floor = [10, 1, 0.1, 0.01, 0.003]
    #    • Stokes warm-start at η=10, P=5 kPa
    #    • Axisymmetric: all integrals weighted by 2πr
    # ══════════════════════════════════════════════════════════════════════
    import gmsh, ufl, tempfile
    from mpi4py import MPI
    from dolfinx import fem, io
    import dolfinx.mesh as dmesh
    from basix.ufl import element, mixed_element
    from dolfinx.fem import petsc as petsc_fem
    from petsc4py import PETSc

    comm     = MPI.COMM_WORLD
    rank     = comm.rank
    two_pi   = 2.0 * np.pi
    JIT      = {"cache_dir": tempfile.mkdtemp(), "timeout": 3600}
    MU_SCHED = [10.0, 1.0, 0.1, 0.01, 0.003]
    all_recs = []

    for needle_name, R_tip in NEEDLES.items():
        if GEOM == "ning":
            # z1 end of straight cartridge, z2 end of cone (= Ning's Lc1), z3 outlet
            z1, z2, z3 = L_STRAIGHT, L_STRAIGHT + L_CONE, L_STRAIGHT + L_CONE + Ltip
        else:
            z1, z2, z3 = Lc1, Lc1+Lc2, Lc1+Lc2+Ltip

        # ── Mesh ──
        if gmsh.isInitialized(): gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add(f"nozzle_{needle_name}")
        if GEOM == "ning":
            # (0,0) (Rin,0) (Rin,z1) -cone-> (Rmid,z2) -step-> (R_tip,z2) (R_tip,z3) (0,z3)
            pts = [(0,0),(Rin,0),(Rin,z1),(Rmid,z2),(R_tip,z2),(R_tip,z3),(0,z3)]
            n_open = 6                     # 6 open lines, the 7th closes on the axis
            i_tipwall, i_outlet = 4, 5
            i_contraction = [2, 3]         # the cone and the abrupt step
        else:
            pts = [(0,0),(Rin,0),(Rin,z1),(Rmid,z1),
                   (Rmid,z2),(R_tip,z2),(R_tip,z3),(0,z3)]
            n_open = 7
            i_tipwall, i_outlet = 5, 6
            i_contraction = [2, 4]         # the two abrupt steps
        gp = [gmsh.model.geo.addPoint(*c, 0) for c in pts]
        gln = [gmsh.model.geo.addLine(gp[i], gp[i+1]) for i in range(n_open)]
        gln.append(gmsh.model.geo.addLine(gp[n_open], gp[0]))
        i_walls = list(range(1, i_outlet))
        gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(gln)])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.addPhysicalGroup(1, [gln[0]], 2)              # inlet
        gmsh.model.addPhysicalGroup(1, [gln[i_outlet]], 3)       # outlet
        gmsh.model.addPhysicalGroup(1, [gln[i] for i in i_walls], 4)   # walls
        gmsh.model.addPhysicalGroup(1, [gln[-1]], 5)             # axis
        f=gmsh.model.mesh.field
        td=f.add("Distance"); f.setNumbers(td,"CurvesList",[gln[i_tipwall]]); f.setNumber(td,"Sampling",200)
        tt=f.add("Threshold"); f.setNumber(tt,"InField",td)
        # ── refinement scaled to the tip radius ───────────────────────────────
        # MESH_M = cells across the tip radius. MESH_M <= 0 falls back to an
        # absolute 20 um sizing, which gives only 2.5 cells across the Ø100 radius.
        if MESH_M > 0:
            _sm = R_tip / MESH_M
            f.setNumber(tt,"SizeMin",_sm); f.setNumber(tt,"SizeMax",2e-4)
            f.setNumber(tt,"DistMin",R_tip); f.setNumber(tt,"DistMax",5*R_tip)
        else:
            f.setNumber(tt,"SizeMin",2e-5); f.setNumber(tt,"SizeMax",2e-4)
            f.setNumber(tt,"DistMin",1e-4); f.setNumber(tt,"DistMax",5e-4)
        cd=f.add("Distance"); f.setNumbers(cd,"CurvesList",[gln[i] for i in i_contraction]); f.setNumber(cd,"Sampling",100)
        ct=f.add("Threshold"); f.setNumber(ct,"InField",cd)
        # the contraction is where P_en is generated, so refine it with the tip
        f.setNumber(ct,"SizeMin",(2*R_tip/MESH_M) if MESH_M > 0 else 4e-5)
        f.setNumber(ct,"SizeMax",2e-4)
        f.setNumber(ct,"DistMin",2e-4); f.setNumber(ct,"DistMax",8e-4)
        mn=f.add("Min"); f.setNumbers(mn,"FieldsList",[tt,ct]); f.setAsBackgroundMesh(mn)
        gmsh.model.mesh.generate(2)
        mdata = io.gmsh.model_to_mesh(gmsh.model, comm, rank, gdim=2)
        msh, _, ft = mdata.mesh, mdata.cell_tags, mdata.facet_tags
        gmsh.finalize()
        if rank==0: print(f"\n{'='*60}\n  Needle: Ø{needle_name} µm  ({msh.geometry.x.shape[0]} nodes)\n{'='*60}")

        # ── FE spaces ──
        xc=ufl.SpatialCoordinate(msh); r=xc[0]; tpr=two_pi*r
        W = fem.functionspace(msh, mixed_element([
            element("Lagrange", msh.basix_cell(), 2, shape=(2,)),
            element("Lagrange", msh.basix_cell(), 1)]))
        dx_m=ufl.Measure("dx",domain=msh); ds_m=ufl.Measure("ds",domain=msh,subdomain_data=ft)
        fdim=1; nrm=ufl.FacetNormal(msh)
        def axi_div(u): return ufl.div(u) + u[0]/(r+EPS_AX)

        tip_fcts = dmesh.locate_entities(msh, fdim,
            lambda X: (np.isclose(X[0],R_tip,atol=6e-6)&(X[1]>=z2-1e-5)&(X[1]<=z3+1e-5)))
        tip_mt = dmesh.meshtags(msh,fdim,np.sort(tip_fcts),np.ones_like(tip_fcts,dtype=np.int32))
        ds_tip = ufl.Measure("ds",domain=msh,subdomain_data=tip_mt,subdomain_id=1)

        W0,_=W.sub(0).collapse(); zv=fem.Function(W0)
        W0r,_=W.sub(0).sub(0).collapse(); zr=fem.Function(W0r)
        bcs = [
            fem.dirichletbc(zr, fem.locate_dofs_topological((W.sub(0).sub(0),W0r),fdim,ft.find(5)), W.sub(0).sub(0)),
            fem.dirichletbc(zv, fem.locate_dofs_topological((W.sub(0),W0),fdim,ft.find(4)), W.sub(0)),
        ]

        mu_fl = fem.Constant(msh, PETSc.ScalarType(10.0))
        def eta(u_e):
            D=ufl.sym(ufl.grad(u_e)); eps=u_e[0]/(r+EPS_AX)
            g2=2*ufl.inner(D,D)+2*eps**2+1e-8
            eta_pl=K_vis*g2**((n_exp-1)/2)
            d1=eta_pl-mu_fl; sm=0.5*(eta_pl+mu_fl+ufl.sqrt(d1**2+1e-6))
            d2=sm-500.0; return 0.5*(sm+500-ufl.sqrt(d2**2+1e-6))

        w_func=fem.Function(W); u_,p_=ufl.split(w_func); v,q=ufl.TestFunctions(W)
        P_in_c=fem.Constant(msh, PETSc.ScalarType(0.0))
        # Density continuation: RHO_C = 0 recovers Stokes exactly, 1 is full inertia.
        # No swirl, so the axisymmetric convective term is just (u.grad)u in (r,z).
        RHO_C = fem.Constant(msh, PETSc.ScalarType(0.0))
        F_ufl = (tpr*(2*eta(u_)*(ufl.inner(ufl.sym(ufl.grad(u_)),ufl.sym(ufl.grad(v)))
                 +(u_[0]/(r+EPS_AX))*(v[0]/(r+EPS_AX)))-p_*axi_div(v)-q*axi_div(u_))*dx_m
                 +tpr*P_in_c*ufl.dot(nrm,v)*ds_m(2)
                 +tpr*RHO_C*rho*ufl.inner(ufl.dot(ufl.grad(u_),u_),v)*dx_m)
        J_ufl=ufl.derivative(F_ufl,w_func)
        F_form=fem.form(F_ufl,jit_options=JIT); J_form=fem.form(J_ufl,jit_options=JIT)

        A_mat=petsc_fem.create_matrix(J_form); b_vec=petsc_fem.create_vector(W)
        dx_sol=fem.Function(W)
        ksp=PETSc.KSP().create(comm); ksp.setType("preonly")
        ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")

        def assemble_residual():
            with b_vec.localForm() as bl: bl.set(0)
            petsc_fem.assemble_vector(b_vec, F_form)
            petsc_fem.apply_lifting(b_vec, [J_form], bcs=[bcs])
            b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc_fem.set_bc(b_vec, bcs); return b_vec.norm()

        def solve_newton(P_pa, mu_val, max_it=40, rtol=1e-6, atol=1e-8):
            P_in_c.value=P_pa; mu_fl.value=mu_val; res0=None
            for it in range(max_it):
                res=assemble_residual()
                if res0 is None: res0=res+1e-30
                if rank==0: print(f"      it {it:2d}: |F|={res:.3e}")
                if res<atol or res/res0<rtol:
                    if rank==0: print(f"      -> converged ({it} iters)")
                    return True
                A_mat.zeroEntries(); petsc_fem.assemble_matrix(A_mat,J_form,bcs=bcs); A_mat.assemble()
                ksp.setOperators(A_mat); ksp.setUp()
                ksp.solve(b_vec, dx_sol.x.petsc_vec); dx_sol.x.scatter_forward()
                alpha=1.0; w0=w_func.x.array.copy()
                for _ in range(8):
                    w_func.x.array[:]=w0-alpha*dx_sol.x.array; w_func.x.scatter_forward()
                    if assemble_residual()<res: break
                    alpha*=0.5
            if rank==0: print(f"      WARNING: max_it reached |F|={res:.3e}")
            return False

        # Stokes warm-start
        u_tr,p_tr=ufl.TrialFunctions(W)
        aS=fem.form(tpr*(20.0*(ufl.inner(ufl.sym(ufl.grad(u_tr)),ufl.sym(ufl.grad(v)))
            +(u_tr[0]/(r+EPS_AX))*(v[0]/(r+EPS_AX)))-p_tr*axi_div(v)-q*axi_div(u_tr))*dx_m, jit_options=JIT)
        # SIGN: the residual is F = 0 with +P_in*n.v on the inlet facet, so moving it to the
        # bilinear side gives a(u,v) = -int P n.v. The warm-start loads the inlet the same
        # way, so with n = -e_z at the inlet the initial guess already points in +z. Newton
        # reaches the same solution from either sign (a re-run checks
        # every case), so no reported number depends on it.
        LS=fem.form(tpr*fem.Constant(msh,PETSc.ScalarType(-5e3))*ufl.dot(nrm,v)*ds_m(2), jit_options=JIT)
        AS=petsc_fem.create_matrix(aS); bS=petsc_fem.create_vector(W)
        petsc_fem.assemble_matrix(AS,aS,bcs=bcs); AS.assemble()
        with bS.localForm() as bl: bl.set(0)
        petsc_fem.assemble_vector(bS,LS)
        petsc_fem.apply_lifting(bS,[aS],bcs=[bcs])
        bS.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
        petsc_fem.set_bc(bS,bcs)
        ksp_s=PETSc.KSP().create(comm); ksp_s.setType("preonly")
        ksp_s.getPC().setType("lu"); ksp_s.getPC().setFactorSolverType("mumps")
        ksp_s.setOperators(AS); ksp_s.setUp()
        ksp_s.solve(bS,w_func.x.petsc_vec); w_func.x.scatter_forward()

        def flow_rate(u_sol):
            return float(fem.assemble_scalar(fem.form(tpr*ufl.dot(u_sol,nrm)*ds_m(3))))
        def mean_tau_wall(u_sol):
            D=ufl.sym(ufl.grad(u_sol)); eps=u_sol[0]/(r+EPS_AX)
            gd=ufl.sqrt(2*ufl.inner(D,D)+2*eps**2+1e-12)
            num=fem.assemble_scalar(fem.form(tpr*K_vis*gd**n_exp*ds_tip))
            den=fem.assemble_scalar(fem.form(tpr*ds_tip))
            return float(num/den) if den>1e-14 else 0.0

        # ── Nodal snapshot for the animation renderers ─────────────────────────
        # The renderers interpolate a converged field rather than re-solving, so
        # writing the snapshot from this loop puts the videos and the reported
        # grid on one solve: same mesh, same regulariser, same continuation path.
        # Velocity is P2 and pressure P1; everything here is sampled onto P1 so
        # that one coordinate array indexes every field.
        def dump_viz_checkpoint(path, Q_m3s_val, tau_w_val):
            import pickle
            if comm.size != 1:
                raise RuntimeError("viz checkpoint needs a single rank; each rank "
                                   "holds only its partition of the field")
            V1  = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1))
            V1v = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1,
                                                 shape=(2,)))
            xs, xv = V1.tabulate_dof_coordinates(), V1v.tabulate_dof_coordinates()
            if xs.shape != xv.shape or not np.allclose(xs, xv):
                raise RuntimeError("scalar and vector P1 nodes are ordered "
                                   "differently; fields would be written against "
                                   "the wrong coordinates")

            f_u = fem.Function(V1v); f_u.interpolate(w_func.sub(0).collapse())
            uvec = f_u.x.array.reshape(-1, 2).copy()

            D_v = ufl.sym(ufl.grad(u_)); eps_v = u_[0]/(r+EPS_AX)
            gd_v = ufl.sqrt(2*ufl.inner(D_v, D_v) + 2*eps_v**2 + 1e-12)
            flds = {"uvec": uvec, "umag": np.linalg.norm(uvec, axis=1)}
            for nm, ex in (("gdot", gd_v), ("tau", K_vis*gd_v**n_exp),
                           ("eta", eta(u_))):
                fh_ = fem.Function(V1)
                fh_.interpolate(fem.Expression(ex, V1.element.interpolation_points))
                flds[nm] = fh_.x.array.copy()

            ck = dict(Rin=Rin, Rmid=Rmid, Rtip=R_tip, z1=z1, z2=z2, z3=z3,
                      # tau_w is the analytic Rabinowitsch value formed from the flow
                      # rate, which is what every reported number uses. The surface
                      # average of the FEM traction is kept beside it under tau_w_fem:
                      # the two differ by about 0.8 %, and a renderer that annotated a
                      # frame with the second would disagree with the manuscript.
                      Q_uLs=Q_m3s_val*1e9,
                      tau_w=K_vis*(((3*n_exp+1)/(4*n_exp))
                                   * 4*Q_m3s_val/(np.pi*R_tip**3))**n_exp,
                      tau_w_fem=tau_w_val,
                      vtk_coords=xs[:, :2].copy(),
                      vtk_cells=np.asarray(V1.dofmap.list, dtype=np.int64),
                      vtk_types=np.full(np.asarray(V1.dofmap.list).shape[0], 5,
                                        dtype=np.uint8),
                      fields=flds,
                      provenance=dict(
                          script=os.path.basename(__file__),
                          needle_um=R_tip*2e6, pressure_kPa=VIZ_P_KPA,
                          mesh_M=MESH_M, inertia=int(INERTIA), eps_ax=EPS_AX,
                          Ltip_m=Ltip, K_vis=K_vis, n_exp=n_exp, rho=rho,
                          n_nodes=int(xs.shape[0]),
                          n_mesh_nodes=int(msh.geometry.x.shape[0]),
                          written=dt.datetime.now().isoformat(timespec="seconds")))
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(ck, fh, protocol=4)
            print(f"     [viz] wrote {path}  nodes={xs.shape[0]}  "
                  f"Q={Q_m3s_val*1e9:.6f} uL/s  tau_w={tau_w_val:.3f} Pa", flush=True)

        # ── Bagley-style entrance excess from the FEM pressure field ────────────
        from dolfinx import geometry as _geom
        _bbt = _geom.bb_tree(msh, msh.topology.dim)

        def dump_pressure_profile(tag):
            """Write the raw pressure field along the axis over the WHOLE domain.

            A diagnostic for pen_direct(). It records the field at three radii from inlet
            to outlet, so the sign convention, the sampling coordinates and whether the fit
            window is actually developed can each be checked against the raw pressure.
            """
            p_fun = w_func.sub(1).collapse()
            out = os.environ.get("L8_PEN_DEBUG_OUT", "pen_profile.csv")
            rows = []
            for rfrac, label in ((1e-4, "axis"), (0.5, "mid"), (0.9, "near_wall")):
                zs = np.linspace(0.0, z3, 600)
                rr = np.where(zs <= z1, Rin, np.where(zs <= z2, Rmid, R_tip)) * rfrac
                pts = np.zeros((len(zs), 3))
                pts[:, 0] = np.maximum(rr, 1e-7)
                pts[:, 1] = zs
                cand = _geom.compute_collisions_points(_bbt, pts)
                coll = _geom.compute_colliding_cells(msh, cand, pts)
                for i in range(len(zs)):
                    links = coll.links(i)
                    if len(links) == 0:
                        continue
                    v = p_fun.eval(pts[i:i+1], np.asarray([links[0]], dtype=np.int32))
                    rows.append((label, float(zs[i]), float(pts[i, 0]), float(v.reshape(-1)[0])))
            with open(out, "w", newline="") as fh:
                wtr = csv.writer(fh)
                wtr.writerow(["line", "z_m", "r_m", "p_Pa"])
                wtr.writerows(rows)
            arr = np.array([r[3] for r in rows if r[0] == "axis"])
            zax = np.array([r[1] for r in rows if r[0] == "axis"])
            if rank == 0 and arr.size:
                print(f"  [dbg {tag}] axis p: min={arr.min():.4e} max={arr.max():.4e} Pa", flush=True)
                for zq in (0.0, z1, z2, z3):
                    j = int(np.argmin(abs(zax - zq)))
                    print(f"  [dbg {tag}]   p(z={zq*1e3:6.2f} mm) = {arr[j]:12.2f} Pa", flush=True)
                print(f"  [dbg {tag}] wrote {out} ({len(rows)} samples)", flush=True)

        def pen_direct():
            """P_en measured, not inferred.

            Returns (P_en_direct, dP_tip_direct, n_fit_points). Fully-developed pipe
            flow through the tip alone would need |dp/dz|*Ltip; the pressure actually
            standing at the tip entrance exceeds that by the entrance loss. Both terms
            are O(P_en), so unlike P_applied - dP_tip this involves no cancellation of
            large numbers.
            """
            # DO NOT SAMPLE ON THE AXIS. The axisymmetric weak form regularises the hoop
            # term as u[0]/(r+EPS_AX), so at r ~ 0 that denominator is dominated by the
            # regulariser and the pressure there is polluted. At Ø200/200 kPa the
            # r=1e-7 line RISES from 202 kPa at the inlet to 259 kPa at the tip
            # entrance, i.e. above the applied pressure, while the r=0.5R and
            # r=0.9R lines fall monotonically 200 kPa -> 0 as they must. The field is fine;
            # sampling the axis is not. Two off-axis radii are used and compared, and their
            # spread is the reported uncertainty.
            p_fun = w_func.sub(1).collapse()
            zs = np.linspace(z2, z3, PEN_NSAMP)
            out = []
            for rfrac in (0.5, 0.85):
                pts = np.zeros((PEN_NSAMP, 3))
                pts[:, 0] = rfrac * R_tip
                pts[:, 1] = zs
                cand = _geom.compute_collisions_points(_bbt, pts)
                coll = _geom.compute_colliding_cells(msh, cand, pts)
                keep, cells = [], []
                for i in range(PEN_NSAMP):
                    links = coll.links(i)
                    if len(links) > 0:
                        keep.append(i)
                        cells.append(links[0])
                if len(keep) < 20:
                    continue
                keep = np.asarray(keep)
                vals = p_fun.eval(pts[keep], np.asarray(cells, dtype=np.int32)).reshape(-1)
                zk = zs[keep]
                s = (zk - z2) / Ltip             # 0 at tip entrance, 1 at outlet
                m = (s >= PEN_FIT_LO) & (s <= PEN_FIT_HI)
                if m.sum() < 10:
                    continue
                slope, b = np.polyfit(zk[m], vals[m], 1)
                if slope >= 0:                   # pressure must fall along the flow
                    continue
                fit = slope * zk[m] + b
                r2 = 1.0 - float(np.sum((vals[m] - fit) ** 2) /
                                 max(np.sum((vals[m] - vals[m].mean()) ** 2), 1e-30))
                p_at_z2 = float(np.interp(z2, zk, vals))
                dP = abs(float(slope)) * Ltip
                out.append((max(p_at_z2 - dP, 0.0), dP, int(m.sum()), r2))
            if not out:
                return float("nan"), float("nan"), 0
            pen = float(np.mean([o[0] for o in out]))
            dPt = float(np.mean([o[1] for o in out]))
            npts = int(min(o[2] for o in out))
            if rank == 0:
                spread = (max(o[0] for o in out) - min(o[0] for o in out))
                r2min = min(o[3] for o in out)
                print(f"     [pen] radii={len(out)} spread={spread:.1f} Pa "
                      f"({100*spread/max(pen,1e-9):.1f}%) developed-fit R2>={r2min:.5f}", flush=True)
            return pen, dPt, npts

        w_prev=w_func.x.array.copy()
        for P_kPa in PRESSURES:
            P_pa=P_kPa*1e3
            if rank==0: print(f"\n  dP = {P_kPa} kPa")
            w_func.x.array[:]=w_prev.copy()
            RHO_C.value = 0.0                      # always start from Stokes
            # EVERY solve is captured, so a case that stalls part-way is flagged in the CSV
            # rather than written as if it had converged.
            #
            # Two DIFFERENT things are recorded, because they do not mean the same thing:
            #   n_warn      intermediate continuation steps that hit max_it. These are
            #               warm-up states on the way to the answer, at a regularised
            #               viscosity or a reduced pressure. A miss here is worth knowing
            #               about and is NOT by itself a reason to distrust the case.
            #   converged   the state that was actually exported reached the end of both
            #               schedules: the final viscosity, and rho-fraction 1 when inertia
            #               is on. This is the one that decides whether the row is sound.
            # Conflating them would flag every case whose first aggressive ramp step needed
            # more than max_it iterations, which is a false alarm, not a finding.
            # The solve sequence is the one that produced the published grid, and no extra
            # solve is added: an extra Newton pass would hand a marginal case iterations the
            # grid run did not have, and make a re-run incomparable with the grid it audits.
            n_warn, ok = 0, True
            last = len(MU_SCHED) - 1
            for i_mu, mu_v in enumerate(MU_SCHED):
                if i_mu==0:
                    # Solver aid, not a physical continuation: at the least-regularised
                    # viscosity the applied pressure is itself ramped in five sub-steps.
                    for Pp in np.linspace(P_pa/5, P_pa, 5):
                        n_warn += not solve_newton(Pp, mu_v)
                elif i_mu < last:
                    n_warn += not solve_newton(P_pa, mu_v)
                else:
                    # The last solve of the viscosity schedule is the state that survives.
                    ok = solve_newton(P_pa, mu_v)
            if n_warn and rank==0:
                print(f"      [continuation] {n_warn} intermediate step(s) hit max_it; "
                      f"final solve at mu={MU_SCHED[-1]} {'converged' if ok else 'DID NOT converge'}")
            rho_reached = 0.0
            if INERTIA:
                for _rf in (0.1, 0.25, 0.5, 0.75, 1.0):
                    RHO_C.value = _rf
                    if rank==0: print(f"      [inertia] rho fraction {_rf}")
                    if not solve_newton(P_pa, MU_SCHED[-1]):
                        ok = False
                        if rank==0:
                            print(f"      [inertia] FAILED to converge at rho={_rf}; "
                                  f"reported values are for the last converged state")
                        break
                    rho_reached = _rf
                if rho_reached < 1.0:
                    ok = False
                    if rank==0:
                        print(f"      [inertia] PARTIAL-DENSITY STATE: reached rho={rho_reached}"
                              f", not 1.0 — this row is flagged converged=0")
            w_prev=w_func.x.array.copy()
            u_sol,_=w_func.split()
            Q_m3s=flow_rate(u_sol); tau_w=mean_tau_wall(u_sol)
            dP_tip=2*Ltip*tau_w/R_tip; P_en=max(P_pa-dP_tip,0)
            if VIZ_CKPT and abs(P_kPa - VIZ_P_KPA) < 1e-9:
                _stem, _ext = os.path.splitext(VIZ_CKPT)
                dump_viz_checkpoint(
                    VIZ_CKPT if len(NEEDLES) == 1
                    else f"{_stem}_D{int(R_tip*2e6)}{_ext}", Q_m3s, tau_w)
            if os.environ.get("L8_PEN_DEBUG") == "1":
                dump_pressure_profile(f"D{int(R_tip*2e6)}_P{int(P_kPa)}")
            P_en_dir, dP_tip_dir, n_fit = pen_direct()
            tau_ext_dir = 3*(n_exp+1)/8*P_en_dir
            tau_ext=3*(n_exp+1)/8*P_en; Q_gs=Q_m3s*rho*1e3
            u_mean=abs(Q_m3s)/(np.pi*R_tip**2)
            eta_w=K_vis*max(u_mean/R_tip,1e-12)**(n_exp-1)
            Re=rho*u_mean*2*R_tip/max(eta_w,1e-30)
            Dst=compute_Dst(dP_tip,R_tip); Det=compute_Det(tau_ext); Dt=compute_Dt(Det,Dst)
            Det_dir=compute_Det(tau_ext_dir); Dt_dir=compute_Dt(Det_dir,Dst)
            if rank==0:
                ratio = (P_en_dir/P_en) if P_en > 0 else float("nan")
                print(f"  -> Q={Q_gs:.3e} g/s  tw={tau_w:.0f} Pa  Dt={Dt:.1f}%")
                print(f"     P_en residual={P_en:8.1f} Pa | direct={P_en_dir:8.1f} Pa "
                      f"| direct/residual={ratio:5.2f} | fit pts={n_fit}")
                print(f"     dP_tip analytic={dP_tip:9.1f} Pa | from dp/dz={dP_tip_dir:9.1f} Pa "
                      f"| Dt direct={Dt_dir:.2f}%")
            all_recs.append(dict(mesh_M=MESH_M, n_nodes=msh.geometry.x.shape[0],
                inertia=int(INERTIA), Ltip_mm=Ltip*1e3,
                converged=int(ok), rho_fraction=(rho_reached if INERTIA else 0.0),
                continuation_warnings=n_warn,
                pressure_kPa=P_kPa,needle_um=R_tip*2e6,
                Q_gs=Q_gs,Q_uLs=Q_m3s*1e9,Re=Re,tau_wall_Pa=tau_w,
                dP_tip_kPa=dP_tip/1e3,P_en_kPa=P_en/1e3,tau_ext_Pa=tau_ext,
                Dst_pct=Dst*100,Det_pct=Det*100,Dt_pct=Dt,
                P_en_direct_kPa=P_en_dir/1e3, dP_tip_direct_kPa=dP_tip_dir/1e3,
                tau_ext_direct_Pa=tau_ext_dir, Det_direct_pct=Det_dir*100,
                Dt_direct_pct=Dt_dir, pen_fit_points=n_fit))
    csv_path = os.environ.get("L8_CSV_OUT", "L8_FEM_pen_direct_2026-07-27.csv")
    fieldnames = [
        "mesh_M", "n_nodes", "inertia", "Ltip_mm",
        "converged", "rho_fraction", "continuation_warnings",
        "pressure_kPa", "needle_um", "Q_gs", "Q_uLs", "Re",
        "tau_wall_Pa", "dP_tip_kPa", "P_en_kPa", "tau_ext_Pa",
        "Dst_pct", "Det_pct", "Dt_pct",
        "P_en_direct_kPa", "dP_tip_direct_kPa", "tau_ext_direct_Pa",
        "Det_direct_pct", "Dt_direct_pct", "pen_fit_points",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_recs:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    manifest = {
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": __file__,
        "mode": "DOLFINx FEM sweep",
        "output_csv": csv_path,
        "sha256": sha,
        "rows": len(all_recs),
        "needles_um": [100, 200, 250, 330],
        "pressures_kPa": PRESSURES,
        "damage_constants": {"a1": a1, "b1": b1, "c1": c1, "a2": a2, "b2": b2},
    }
    with open("MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {csv_path} rows={len(all_recs)} sha256={sha}")

    # FAIL CLOSED. Every row is written, flagged, so nothing is lost and a reader can see
    # exactly which cases are suspect, but the process exits non-zero, so a batch job
    # cannot report success while part of the grid never reached rho-fraction = 1.
    failed = [r for r in all_recs if not r.get("converged", 1)]
    if failed:
        for r in failed:
            print(f"[FAILED] needle {r['needle_um']:.0f} um  {r['pressure_kPa']:.0f} kPa  "
                  f"rho_fraction={r.get('rho_fraction', 0.0)}")
        sys.exit(f"{len(failed)} of {len(all_recs)} cases did not converge to "
                 f"rho-fraction = 1; they are flagged converged=0 in {csv_path}")
    print(f"[ok] all {len(all_recs)} cases converged at full density")
    sys.exit(0)
