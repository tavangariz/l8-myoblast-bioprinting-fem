#!/usr/bin/env python3
"""Test of assert_labels_clear: it must fail on merged tick labels and on a legend wider than its panel.

    python scripts/tests/test_labels_clear.py
"""
import pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import l8_style as S


def expect_fail(name, build):
    fig, ax = plt.subplots(figsize=(1.4, 1.6), dpi=400)
    build(ax)
    try:
        S.assert_labels_clear(fig, [ax])
    except AssertionError as e:
        print(f"  PASS  {name}: caught -- {str(e).splitlines()[-1].strip()}")
        return 0
    finally:
        plt.close(fig)
    print(f"  FAIL  {name}: the gate did not fire")
    return 1


def merged_tick_labels(ax):
    ax.bar(range(4), [1, 2, 3, 4])
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Ø{d}" for d in (100, 200, 250, 330)])
    ax.tick_params(labelsize=9)


def key_leaves_panel(ax):
    for i, d in enumerate((100, 200, 250, 330)):
        ax.plot([1, 2], [i, i + 1], label=f"Ø{d} µm — a deliberately long entry")
    ax.legend(fontsize=9, loc="upper left")


if __name__ == "__main__":
    rc = expect_fail("tick labels merged", merged_tick_labels)
    rc += expect_fail("key leaves its panel", key_leaves_panel)
    print("all gates fired" if rc == 0 else "GATE DID NOT FIRE")
    sys.exit(rc)
