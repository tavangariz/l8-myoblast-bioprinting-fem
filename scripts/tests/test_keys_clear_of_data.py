#!/usr/bin/env python3
"""Test of assert_keys_clear_of_data: a legend over a curve must stop the figure being written, including a curve with no vertex inside the legend, and a legend below the axes must pass.

    python scripts/tests/test_keys_clear_of_data.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l8_style as S  # noqa: E402


def _panel(where):
    fig, ax = plt.subplots(figsize=(4, 3))
    # Five points only: the segments are long, the vertices are not in the key
    ax.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], "o-", label="a line that crosses the panel")
    if where == "inside":
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.35)
    return fig


def main() -> int:
    ok = True

    fig = _panel("inside")
    hits = S.keys_over_data(fig)
    vertices_inside = 0
    lb = fig.axes[0].get_legend().get_window_extent(fig.canvas.get_renderer())
    for x, y in fig.axes[0].transData.transform(fig.axes[0].lines[0].get_xydata()):
        vertices_inside += (lb.x0 <= x <= lb.x1 and lb.y0 <= y <= lb.y1)
    if hits:
        print(f"planted defect: caught, {hits[0][2]} sampled points under the key "
              f"(and {vertices_inside} of the 5 vertices, which is why sampling is needed)")
    else:
        print("planted defect: NOT CAUGHT -- the checker is broken")
        ok = False
    try:
        S.assert_keys_clear_of_data(fig)
        print("planted defect: assert_keys_clear_of_data did NOT raise -- broken")
        ok = False
    except AssertionError:
        print("planted defect: assert_keys_clear_of_data raised, as it must")
    plt.close(fig)

    fig = _panel("below")
    if S.keys_over_data(fig):
        print("clean panel: FALSE POSITIVE -- a key below the axes was reported as covering data")
        ok = False
    else:
        print("clean panel: passes, as it must")
    S.assert_keys_clear_of_data(fig)
    plt.close(fig)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
