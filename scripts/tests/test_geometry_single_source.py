#!/usr/bin/env python3
"""The nozzle geometry is stated once, in `l8_geometry`, and every live script reads it there.

Two checks:
  1. `GEOM`, `L_NEEDLE` and `TIP_RADIUS_M` carry the dimensions of Ning et al. 2018, Fig. 1
     (cartridge radius 2.1 mm, mid-section radius 1.0 mm, reference tip radius 100 um,
     z1 10.5 mm, z2 12.0 mm, z3 22.0 mm, needle length 10 mm; tips of 100, 200, 250, 330 um).
  2. No live script writes one of those values as a literal. Archived solvers (`_legacy/`)
     and the tests are exempt. The scan is first run on a planted restatement, so a scan
     that cannot fire is caught.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from l8_geometry import GEOM, L_NEEDLE, TIP_RADIUS_M  # noqa: E402

LITERAL = re.compile(r"(?<![\w.])(50e-6|100e-6|125e-6|165e-6|2\.1e-3|1\.0e-3|10\.5e-3|"
                     r"12\.0e-3|12e-3|22\.0e-3|22e-3|10e-3|10\.0e-3)(?![\w.])")
EXEMPT_FILES = {"l8_geometry.py"}
EXEMPT_DIRS = {"_legacy", "tests", "__pycache__"}
# Same digits, different quantity. Each entry names what the value is.
ALLOWED = {
    ("fig5_lagrangian_tracers.py", "edge_trim=50e-6"): "axial trim at the needle ends, not a radius",
}


def restatements(text):
    """Lines of code (not comments, not docstring examples) that write a geometry literal."""
    hits, in_doc = [], False
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.count('"""') % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code = s.split("#", 1)[0]
        if LITERAL.search(code):
            hits.append((i, s))
    return hits


def main():
    fails = 0
    want = dict(Rin=2.1e-3, Rmid=1.0e-3, Rtip=100e-6, z1=10.5e-3, z2=12.0e-3, z3=22.0e-3)
    for k, v in want.items():
        if GEOM[k] != v:
            print(f"  FAIL  GEOM[{k!r}] = {GEOM[k]!r}, Ning Fig. 1 gives {v!r}")
            fails += 1
    if L_NEEDLE != 10e-3 or abs(GEOM["z3"] - GEOM["z2"] - L_NEEDLE) > 1e-12:
        print(f"  FAIL  L_NEEDLE = {L_NEEDLE!r} does not match z3 - z2")
        fails += 1
    if TIP_RADIUS_M != {100: 50e-6, 200: 100e-6, 250: 125e-6, 330: 165e-6}:
        print(f"  FAIL  TIP_RADIUS_M = {TIP_RADIUS_M!r}")
        fails += 1
    if not fails:
        print("  PASS  l8_geometry carries Ning's Fig. 1 dimensions")

    planted = 'x = 1\nNEEDLES = {"200": 100e-6}\n# 100e-6 in a comment is fine\n'
    if restatements(planted) != [(2, 'NEEDLES = {"200": 100e-6}')]:
        print("  FAIL  the scan did not catch a planted restatement")
        fails += 1
    else:
        print("  PASS  the scan catches a planted restatement")

    found = []
    for p in sorted(HERE.rglob("*.py")):
        if p.name in EXEMPT_FILES or EXEMPT_DIRS & set(p.relative_to(HERE).parts):
            continue
        rel = p.relative_to(HERE)
        found += [(rel, i, s) for i, s in restatements(p.read_text())
                  if not any(str(rel) == f and frag in s for f, frag in ALLOWED)]
    for p, i, s in found:
        print(f"  FAIL  {p}:{i}: {s[:100]}")
    if found:
        fails += len(found)
    else:
        print("  PASS  no live script restates the geometry")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
