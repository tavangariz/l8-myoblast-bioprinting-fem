#!/usr/bin/env python3
"""Test of keys_over_data on contour plots: a legend over a contour line must be reported, and a legend below the axes must not, even when the data origin falls inside it.

    python scripts/tests/test_keys_clear_of_contours.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l8_style as S  # noqa: E402


def _field(ax, x0, x1, y0, y1):
    X, Y = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60))
    Z = (X - x0) / (x1 - x0) + (Y - y0) / (y1 - y0)
    ax.contourf(X, Y, Z, levels=12)
    ax.contour(X, Y, Z, levels=[1.4], colors="white")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def main() -> int:
    ok = True

    # Planted: the Z = 0.6 line crosses the upper left, where the key sits
    fig, ax = plt.subplots(figsize=(4, 3))
    X, Y = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
    ax.contour(X, Y, Y - X, levels=[0.6], colors="k")
    ax.legend(handles=[plt.Line2D([], [], color="k", label="a contour under the key")],
              loc="upper left")
    hits = S.keys_over_data(fig)
    if hits:
        print(f"planted contour under the key: caught, {hits[0][2]} vertices")
    else:
        print("planted contour under the key: NOT CAUGHT -- the checker is broken")
        ok = False
    plt.close(fig)

    # Clean: key below a filled field, over the screen position of the data origin
    fig, ax = plt.subplots(figsize=(6, 4))
    _field(ax, 25, 400, 100, 330)
    fig.subplots_adjust(left=0.30, bottom=0.45)
    leg = ax.legend(handles=[plt.Line2D([], [], color="k", label=f"entry {i}") for i in range(6)],
                    loc="upper left", bbox_to_anchor=(-0.35, -0.12), ncol=2)
    fig.canvas.draw()
    lb = leg.get_window_extent(fig.canvas.get_renderer())
    ox, oy = ax.transData.transform((0.0, 0.0))
    if not (lb.x0 <= ox <= lb.x1 and lb.y0 <= oy <= lb.y1):
        print(f"setup: the data origin ({ox:.0f}, {oy:.0f}) is not inside the key {lb.bounds}; "
              "the clean case does not test what it claims")
        ok = False
    hits = S.keys_over_data(fig)
    if hits:
        print(f"clean panel: FALSE POSITIVE -- {hits}")
        ok = False
    else:
        print("clean panel: key over the data origin, below the axes, passes as it must")
    plt.close(fig)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
