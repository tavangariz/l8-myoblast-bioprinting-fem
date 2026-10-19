#!/usr/bin/env python3
"""Test of the canvas check in save_figure: a panel letter cut by the top edge must fail, and a figure clear of every edge must pass.

    python scripts/tests/test_ink_inside_canvas.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l8_style as S  # noqa: E402


def main():
    # The second case leaves wide pixel margins (top=0.8), so only the text-box check can see
    # a letter drawn wholly above the page.
    for top, dy, what in ((0.97, 0, "straddling the top edge"), (0.8, 60, "wholly above the canvas")):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
        fig.subplots_adjust(top=top, left=0.2, bottom=0.15)   # tick labels on the canvas
        ax.annotate("A", xy=(0, 1), xycoords="axes fraction", xytext=(-10, dy),
                    textcoords="offset points", fontsize=14, fontweight="bold",
                    va="bottom", annotation_clip=False)
        try:
            S.assert_ink_inside_canvas(fig)
        except AssertionError as e:
            print(f"  PASS  letter {what} caught: {e}")
        else:
            raise SystemExit(f"FAIL: a letter {what} passed the guard")
        plt.close(fig)

    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "schematic", ha="center", va="center")
    S.assert_ink_inside_canvas(fig)                   # must not raise
    print("  PASS  full-canvas schematic with clear margins passes")
    plt.close(fig)
    print("all gates fired")


if __name__ == "__main__":
    main()
