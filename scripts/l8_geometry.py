#!/usr/bin/env python3
"""Cartridge and nozzle geometry, after Ning et al. (2018), Fig. 1.

D1 = 4.2 mm, D2 = 2.0 mm, Lc1 = 12 mm (whole cartridge), Lc2 = 10.5 mm (its straight part),
nozzle length 10 mm, nozzle diameters 100-330 µm:

    z = 0        inlet, applied pressure
    z = 10.5 mm  end of the straight cartridge
    z = 12.0 mm  end of the conical transition, nozzle entry
    z = 22.0 mm  nozzle outlet, zero pressure

wall_radius() is the single definition of the wall, used by the figures and the tracer integration.
"""
from __future__ import annotations
import numpy as np

# The domain, in metres. Callers import this rather than restating the numbers, so a script
# that draws the bore and a script that meshes it are working from one definition.
GEOM = {
    "Rin":  2.1e-3,    # D1 / 2, the cartridge bore
    "Rmid": 1.0e-3,    # D2 / 2, the funnel exit
    "Rtip": 100e-6,    # D  / 2 at the O200 um reference tip
    "z1":   10.5e-3,   # L_c2, end of the straight cartridge
    "z2":   12.0e-3,   # L_c1, end of the funnel and start of the needle
    "z3":   22.0e-3,   # L_c1 + L, needle outlet
}

# Tip radii the study sweeps, keyed by tip diameter in micrometres.
TIP_RADIUS_M = {100: 50e-6, 200: 100e-6, 250: 125e-6, 330: 165e-6}


def wall_radius(z, geom=None):
    """Bore radius at axial position z, in the units of `geom`. Vectorised.

    Constant at Rin to z1, a linear taper from Rin to Rmid across the funnel,
    then the needle bore Rtip beyond z2.
    """
    geom = geom or GEOM
    z = np.asarray(z, dtype=float)
    taper = geom["Rin"] + (geom["Rmid"] - geom["Rin"]) * np.clip(
        (z - geom["z1"]) / (geom["z2"] - geom["z1"]), 0.0, 1.0)
    return np.where(z > geom["z2"], geom["Rtip"], taper)


def profile_points(geom=None):
    """(z, r) tracing the bore wall: cartridge, funnel, contraction, needle.

    Five points. The contraction is the pair that repeats z2 at two radii.
    """
    geom = geom or GEOM
    z = np.array([0.0, geom["z1"], geom["z2"], geom["z2"], geom["z3"]])
    r = np.array([geom["Rin"], geom["Rin"], geom["Rmid"],
                  geom["Rtip"], geom["Rtip"]])
    return z, r


def _self_test():
    """The taper is the claim; check it at the ends and in the middle."""
    g = {"Rin": 2.1e-3, "Rmid": 1.0e-3, "Rtip": 100e-6,
         "z1": 10.5e-3, "z2": 12.0e-3, "z3": 22.0e-3}
    checks = [(0.0, 2.1e-3), (10.5e-3, 2.1e-3), (11.25e-3, 1.55e-3),
              (12.0e-3, 1.0e-3), (12.001e-3, 100e-6), (22.0e-3, 100e-6)]
    for z, want in checks:
        got = float(wall_radius(z, g))
        assert abs(got - want) < 1e-9, f"z={z}: got {got}, want {want}"
    zz, rr = profile_points(g)
    assert len(zz) == 5 and zz[2] == zz[3], "the contraction must be one step"
    print("l8_geometry self-test: OK")


if __name__ == "__main__":
    _self_test()
