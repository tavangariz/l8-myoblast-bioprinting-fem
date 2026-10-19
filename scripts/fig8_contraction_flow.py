#!/usr/bin/env python3
"""Fig. 8: flow field at the cartridge-to-nozzle contraction, reference condition (200 kPa, 200 µm).

(A) Shear rate, with streamlines integrated from the solved velocity field. (B) Centerline velocity
and the extensional strain rate du_z/dz it implies. The damage law does not use this local strain
rate: it takes one effective extensional stress per case from the entrance pressure (Eq. 22).

    python scripts/fig8_contraction_flow.py
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l8_style as S  # noqa: E402
from l8_geometry import wall_radius  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures" / "Fig8_contraction_flow.png"
from l8_paths import CHECKPOINT, require  # noqa: E402

CKPT = CHECKPOINT

# Window on the geometry, in metres. z2 is the mid-to-tip step.
Z_BEFORE, Z_AFTER = 1.60e-3, 1.10e-3
# The radial window spans the bore that is actually there at the left edge of the view.
NR, NZ = 420, 620
N_SEED = 11
# Shear rate, s^-1, on a log scale.
CLIM = (1.0, 1.0e5)


def load():
    d = pickle.load(open(require(CKPT, "the solver checkpoint (set L8_CKPT)"), "rb"))
    p = d["provenance"]
    assert p["needle_um"] == 200.0 and p["pressure_kPa"] == 200.0, \
        f"expected the O200/200 kPa reference solve, got {p['needle_um']}/{p['pressure_kPa']}"
    return d


def main():
    d = load()
    plt.rcParams.update(S.rc(S.PAPER))
    c = d["vtk_coords"]
    r, z = c[:, 0], c[:, 1]
    uvec, gdot = d["fields"]["uvec"], d["fields"]["gdot"]
    z2, rtip, rmid = d["z2"], d["Rtip"], d["Rmid"]

    zlo, zhi = z2 - Z_BEFORE, z2 + Z_AFTER
    geom = {"Rin": d["Rin"], "Rmid": rmid, "Rtip": rtip,
            "z1": d["z1"], "z2": z2, "z3": d["z3"]}
    r_top = float(wall_radius(zlo, geom)) * 1.05
    zg = np.linspace(zlo, zhi, NZ)
    rg = np.linspace(0.0, r_top, NR)
    Zg, Rg = np.meshgrid(zg, rg)

    interp_g = LinearNDInterpolator(np.column_stack([z, r]), gdot)
    interp_uz = LinearNDInterpolator(np.column_stack([z, r]), uvec[:, 1])
    interp_ur = LinearNDInterpolator(np.column_stack([z, r]), uvec[:, 0])
    G = interp_g(Zg, Rg)
    UZ = interp_uz(Zg, Rg)
    UR = interp_ur(Zg, Rg)

    # Outside the bore the interpolant is undefined, and because the domain is non-convex
    # LinearNDInterpolator will happily extrapolate across the step rather than return NaN.
    wall = wall_radius(Zg, geom)
    outside = Rg > wall
    for A in (G, UZ, UR):
        A[outside] = np.nan

    # The symmetry axis is clamped to the colour floor, because white means "outside the domain" here.
    _inside = ~outside
    _pre_clamp = G.copy()
    G[_inside] = np.maximum(G[_inside], CLIM[0])

    lo, hi = float(np.nanmin(G)), float(np.nanmax(G))
    # Measured over the domain, not over the whole rectangle: averaging across the masked
    # outside region would dilute this fraction and let it pass regardless of what was clipped.
    clipped = float(np.mean(_pre_clamp[_inside] < CLIM[0])) * 100.0
    assert hi <= CLIM[1], \
        f"shear rate reaches {hi:.4g} s^-1, above the colour ceiling {CLIM[1]:.4g}"
    assert clipped < 3.0, \
        f"{clipped:.2f} % of the window falls below the colour floor; that is too much to clip"

    fig, (axA, axB) = plt.subplots(
        # 170 mm on the page.
        1, 2, figsize=(6.0208, 2.5900), dpi=S.PAPER["dpi"],
        gridspec_kw=dict(width_ratios=[1.28, 1.0], wspace=0.34,
                         left=0.075, right=0.965, bottom=0.185, top=0.885))

    mesh = axA.pcolormesh(Zg * 1e3, Rg * 1e6, G, cmap="cividis",
                          norm=LogNorm(*CLIM), shading="auto", rasterized=True)
    seed_r = np.linspace(0.04, 0.95, N_SEED) * float(wall_radius(zlo + 0.06e-3, geom))
    seeds = np.column_stack([np.full(N_SEED, (zlo + 0.06e-3) * 1e3), seed_r * 1e6])
    # Streamplot integrates in DATA coordinates, and this panel plots z in mm against r in um.
    SZ, SR = UZ.copy(), UR.copy()
    past_step = Zg > z2
    SZ[past_step] = np.nan
    SR[past_step] = np.nan
    axA.streamplot(Zg * 1e3, Rg * 1e6, SZ * 1e3, SR * 1e6, color="white", linewidth=0.85,
                   density=6.0, arrowsize=0.75, start_points=seeds,
                   integration_direction="forward", broken_streamlines=False)
    # The bore outline, drawn rather than filled, as Figure 1 draws the same needle.
    z_out = np.concatenate([np.linspace(zlo, z2, 200), [z2], np.linspace(z2, zhi, 60)])
    r_out = np.concatenate([wall_radius(np.linspace(zlo, z2, 200), geom),
                            [rtip], np.full(60, rtip)])
    axA.plot(z_out * 1e3, r_out * 1e6, color=S.INK, lw=1.4,
             solid_joinstyle="miter", zorder=6)
    axA.set_xlim(zlo * 1e3, zhi * 1e3)
    axA.set_ylim(0, r_top * 1e6)
    axA.set_xlabel(S.LBL["axial_z"])
    axA.set_ylabel(S.LBL["radius_um"])
    S.panel_label(axA, "A")
    axA.grid(False)
    cb = fig.colorbar(mesh, ax=axA, pad=0.022)
    cb.set_label("Shear rate  $\\dot{\\gamma}$  (s$^{-1}$)",
                 fontsize=S.PAPER["label"], fontweight="bold")
    cb.ax.tick_params(labelsize=S.PAPER["tick"])

    uz_axis = interp_uz(zg, np.zeros_like(zg))
    dz = zg[1] - zg[0]
    eps_dot = np.gradient(uz_axis, dz)
    axB.plot(zg * 1e3, uz_axis, color=S.DIA_C[200], lw=S.PAPER["lw_data"],
             label="axial velocity $u_z$")
    axB.set_xlabel(S.LBL["axial_z"])
    axB.set_ylabel("Centerline  $u_z$  (m s$^{-1}$)")
    S.panel_label(axB, "B")
    axB.set_xlim(zlo * 1e3, zhi * 1e3)

    axB2 = axB.twinx()
    axB2.plot(zg * 1e3, eps_dot, color=S.ANNOT, lw=S.PAPER["lw_data"], ls="--",
              label="strain rate $\\mathrm{d}u_z/\\mathrm{d}z$")
    axB2.set_ylabel("Extensional strain rate (s$^{-1}$)",
                    fontsize=S.PAPER["label"], fontweight="bold")
    axB2.tick_params(labelsize=S.PAPER["tick"])
    axB2.grid(False)

    # The step is named in the key, not written on the panel.
    axB.axvline(z2 * 1e3, color=S.INK, lw=1.0, ls=":", zorder=1,
                label="contraction step")

    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    # Upper left, inside the axes.
    for _ax in (axB, axB2):
        _lo, _hi = _ax.get_ylim()
        _ax.set_ylim(_lo, _lo + 1.45 * (_hi - _lo))

    leg_b = axB.legend(h1 + h2, l1 + l2, fontsize=S.PAPER["legend"] - 0.5,
                       loc="upper left", ncol=1, framealpha=1.0,
                       borderaxespad=0.5, handlelength=1.8, labelspacing=0.35)
    leg_b.set_zorder(10)   # the curves are drawn above a default legend and print through it

    # Nothing plotted may fall inside the key.
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _lb = leg_b.get_window_extent(_rend)
    for _ax, _y in ((axB, uz_axis), (axB2, eps_dot)):
        _pts = _ax.transData.transform(np.column_stack([zg * 1e3, _y]))
        _in = ((_pts[:, 0] >= _lb.x0) & (_pts[:, 0] <= _lb.x1)
               & (_pts[:, 1] >= _lb.y0) & (_pts[:, 1] <= _lb.y1))
        assert not _in.any(), (
            f"the key covers {int(_in.sum())} of {len(_pts)} plotted samples on "
            f"{'left' if _ax is axB else 'right'} axis; move it or shrink it")
    print(f"  panel (b) key: upper left, clear of all {2 * len(zg)} plotted samples")

    # Numbers the caption quotes, computed here and printed so they are never typed
    i_pk = int(np.nanargmax(eps_dot))
    pk = float(eps_dot[i_pk])
    half = np.where(eps_dot >= 0.5 * pk)[0]
    width_mm = float((zg[half[-1]] - zg[half[0]]) * 1e3) if len(half) > 1 else float("nan")
    u_in = float(np.nanmean(uz_axis[zg < z2 - 1.0e-3]))
    u_out = float(np.nanmax(uz_axis))
    print(f"  centerline u_z: {u_in:.4f} -> {u_out:.4f} m/s  (x{u_out / u_in:.1f})")
    print(f"  peak du_z/dz  : {pk:.1f} s^-1 at z = {zg[i_pk] * 1e3:.3f} mm "
          f"(step at {z2 * 1e3:.1f} mm)")
    print(f"  full width at half peak: {width_mm:.3f} mm")
    print(f"  shear rate in window   : {lo:.3g} to {hi:.0f} s^-1, {clipped:.2f} % below the floor")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Panel letters snap to each panel's real top-left corner once the layout is fixed.
    S.snap_panel_labels(fig)
    fig.savefig(OUT, dpi=S.PAPER["dpi"], bbox_inches="tight", facecolor="white")
    from l8_paths import flatten_png; flatten_png(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
