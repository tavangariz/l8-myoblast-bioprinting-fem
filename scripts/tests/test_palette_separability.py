#!/usr/bin/env python3
"""Test of the nozzle-diameter palette.

Measures the smallest CIELAB distance between entries in normal vision and under simulated
protanopia, deuteranopia and tritanopia (Viénot, Brettel and Mollon 1999), WCAG contrast against
white, monotonic luminance, and distance from viridis. Exits 1 if the committed palette is not
the best of the candidates.

    python scripts/tests/test_palette_separability.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l8_style as S  # noqa: E402


def hex2rgb(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb): return float(np.dot(_lin(rgb), [0.2126, 0.7152, 0.0722]))


def contrast_on_white(rgb):
    return (1.0 + 0.05) / (luminance(rgb) + 0.05)


def rgb2lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ _lin(rgb) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


# Vienot-Brettel-Mollon 1999: project onto the dichromat's reduced colour plane in LMS.
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.02996, 0.18431, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_SIM = {
    "protanopia":   np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    "deuteranopia": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    "tritanopia":   np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}


def simulate(rgb, kind):
    if kind == "normal":
        return rgb
    out = _LMS2RGB @ (_SIM[kind] @ (_RGB2LMS @ _lin(rgb)))
    out = np.clip(out, 0, 1)
    return np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)


def worst_separation(colors):
    """Smallest CIELAB distance between any pair, over normal vision and all three dichromacies."""
    worst, where = 1e9, None
    for kind in ("normal", "protanopia", "deuteranopia", "tritanopia"):
        labs = [rgb2lab(simulate(c, kind)) for c in colors]
        for (i, a), (j, b) in combinations(list(enumerate(labs)), 2):
            d = float(np.linalg.norm(a - b))
            if d < worst:
                worst, where = d, f"{kind}, entries {i} and {j}"
    return worst, where


def viridis_distance(colors):
    # Matplotlib.colormaps, not cm.get_cmap: the latter is removed in Matplotlib 3.11.
    import matplotlib
    vir = [matplotlib.colormaps["viridis"](t)[:3] for t in np.linspace(0, 1, 64)]
    labs = [rgb2lab(np.array(v)) for v in vir]
    return min(float(np.linalg.norm(rgb2lab(c) - l)) for c in colors for l in labs)


# Colours this paper has already given a meaning to, in two classes, because they are not equally
# dangerous and pretending they are makes the check unsatisfiable.
SERIES = {"shear component": S.COMP_C["shear"],
          "extensional component": S.COMP_C["extensional"],
          "Ning measured squares": "#1A1A1A", "black": "#000000"}

# Furniture: axis ink, de-emphasised text, threshold and reference lines.
FURNITURE = {"reference line (ANNOT)": S.ANNOT, "threshold line": S.THRESHOLD,
             "muted furniture": S.MUTE, "axis ink": S.INK}

RESERVED = {**SERIES, **FURNITURE}
SERIES_FLOOR = 20.0
FURNITURE_FLOOR = 10.0

CANDIDATES = {
    "shipped (whatever l8_style holds now)": [S.DIA_C[d] for d in S.DIA_ORDER],
    "teal->navy": ["#6BC3CC", "#2494A9", "#0D6076", "#0A2C48"],
    "magenta ramp": ["#F1B6DA", "#DE77AE", "#C51B7D", "#8E0152"],
    "Okabe-Ito 4, luminance-ordered": ["#E69F00", "#009E73", "#0072B2", "#000000"],
    "best subset overall, rejected: black is Ning's": ["#E69F00", "#56B4E9", "#0072B2", "#000000"],
    "Okabe-Ito with vermillion": ["#E69F00", "#009E73", "#D55E00", "#0072B2"],
    "Tol vibrant 4": ["#EE7733", "#009988", "#0077BB", "#332288"],
    "Tol high-contrast + teal": ["#DDAA33", "#009988", "#BB5566", "#004488"],
}


# The mesh hue (Fig. 2B) shares the page set with the diameters. It must not read as one of
# them, and it must stand apart from the INK wall drawn over it, under every vision type.
MESH_FLOOR = 20.0


def mesh_distance(hex_):
    """Nearest diameter colour and the wall, each over normal vision and all three dichromacies."""
    c = hex2rgb(hex_)
    groups = {"diameter": {f"Ø{d}": S.DIA_C[d] for d in S.DIA_ORDER}, "wall": {"INK": S.INK}}
    return {g: min((float(np.linalg.norm(rgb2lab(simulate(c, vis)) -
                                         rgb2lab(simulate(hex2rgb(v), vis)))), f"{k}, {vis}")
                   for k, v in cols.items()
                   for vis in ("normal", "protanopia", "deuteranopia", "tritanopia"))
            for g, cols in groups.items()}


def report(name, hexes):
    cols = [hex2rgb(h) for h in hexes]
    sep, where = worst_separation(cols)
    lums = [luminance(c) for c in cols]
    con = [contrast_on_white(c) for c in cols]
    mono = all(lums[i] > lums[i + 1] for i in range(len(lums) - 1))
    vd = viridis_distance(cols)
    # Under every vision type, not just normal vision.
    def _closest(group):
        return min(float(np.linalg.norm(rgb2lab(simulate(c, vis)) -
                                        rgb2lab(simulate(hex2rgb(v), vis))))
                   for c in cols for v in group.values()
                   for vis in ("normal", "protanopia", "deuteranopia", "tritanopia"))
    rd, fd = _closest(SERIES), _closest(FURNITURE)
    print(f"{name}")
    print(f"    worst separation {sep:6.1f}  ({where})")
    print(f"    contrast on white min {min(con):.2f}:1   luminance ordered: "
          f"{'yes' if mono else 'NO'}")
    print(f"    nearest viridis  {vd:6.1f}   nearest other series {rd:5.1f}   "
          f"nearest furniture {fd:5.1f}")
    print(f"    {' '.join(hexes)}")
    return sep, mono, min(con), vd, rd, fd


def main() -> int:
    print("Diameter palette, four series in one panel. Higher separation is better;\n"
          "CIELAB distance under 20 is where two lines start to be confusable.\n")
    scores = {}
    for name, hexes in CANDIDATES.items():
        scores[name] = report(name, hexes)
        print()
    shipped = scores["shipped (whatever l8_style holds now)"]
    best = max(v[0] for k, v in scores.items() if not k.startswith("shipped"))
    ok = True
    if shipped[0] < best - 0.5:
        print(f"FAIL: the shipped palette separates at {shipped[0]:.1f} while a candidate "
              f"reaches {best:.1f}.")
        ok = False
    if not shipped[1]:
        print("FAIL: the shipped palette is not ordered by luminance, so the ordered "
              "category does not survive greyscale.")
        ok = False
    if shipped[2] < 2.0:
        print(f"FAIL: shipped palette has an entry at {shipped[2]:.2f}:1 on white; a thin "
              f"line needs 2:1.")
        ok = False
    # 12, not 25.
    if shipped[3] < 12.0:
        print(f"FAIL: shipped palette sits {shipped[3]:.1f} from viridis, which is the "
              f"D_t ramp; a diameter must not read as a damage value.")
        ok = False
    if shipped[4] < SERIES_FLOOR:
        print(f"FAIL: a shipped diameter colour is {shipped[4]:.1f} from another DATA SERIES "
              f"({', '.join(SERIES)}); two measurements would look alike.")
        ok = False
    if shipped[5] < FURNITURE_FLOOR:
        print(f"FAIL: a shipped diameter colour is {shipped[5]:.1f} from axis furniture "
              f"({', '.join(FURNITURE)}).")
        ok = False
    md = mesh_distance(S.MESH_C)
    print(f"Mesh hue {S.MESH_C}: nearest diameter {md['diameter'][0]:.1f} ({md['diameter'][1]}), "
          f"wall {md['wall'][0]:.1f} ({md['wall'][1]})")
    for g, (d, where) in md.items():
        if d < MESH_FLOOR:
            print(f"FAIL: the mesh hue sits {d:.1f} from the {g} ({where}); the floor is "
                  f"{MESH_FLOOR:.0f}.")
            ok = False
    print("PASS: the shipped diameter palette is the best measured candidate, and the mesh hue "
          "reads as neither a diameter nor the wall." if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
