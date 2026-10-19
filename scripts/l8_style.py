"""Figure style: colours, colormaps, markers, type sizes, and the layout checks run before saving.

Measured quantities use one perceptually uniform colormap each; nozzle diameters use an ordered
four-colour ramp kept distinct from viridis. Two type-size profiles: PAPER for the journal page
and SLIDE for presentations.
"""
from __future__ import annotations

import numpy as np

# Nozzle diameter: an ordered category.
DIA_C: dict[int, str] = {
    100: "#E69F00",   # Okabe-Ito orange
    200: "#56B4E9",   # Okabe-Ito sky blue
    250: "#882255",   # Tol muted wine
    330: "#332288",   # Tol muted indigo, the darkest and so the widest nozzle
}

# One marker per diameter, so identity survives colour vision deficiency, greyscale printing and two
# lines crossing.
DIA_M: dict[int, str] = {100: "o", 200: "s", 250: "^", 330: "D"}

DIA_ORDER = (100, 200, 250, 330)

# Continuous fields. One colormap per quantity. Changing one of these changes every figure
# that plots that quantity, which is the point.
CMAP_DT = "viridis"     # total damage D_t (%), the paper's central quantity
CMAP_SD = "magma"       # GP posterior s.d.: an uncertainty, deliberately NOT the value's ramp.
# An uncertainty is never drawn on viridis, the ramp of the value it qualifies.
CMAP_Q = "plasma"       # volumetric flow rate
CMAP_MESH = "viridis"   # mesh element size
CMAP_TAU = "cividis"    # wall shear stress
CMAP_VEL = "cividis"    # velocity magnitude

# Four quantities, four perceptually uniform ramps, so a multi-panel figure never puts the same colour
# on two different things.

# Annotation. Never encodes anything; only points at things.
INK = "#333333"          # axis furniture, body annotation
ANNOT = "#B4501F"        # burnt orange: reference lines and their labels. Outside viridis
                         # and dark enough to hold at 7 pt.
MUTE = "#6B6B6B"         # secondary text, walls, de-emphasis
RULE = "#CCCCCC"         # hairlines, box edges
PANEL = "#F2F2F4"        # pale panel fill, e.g. the bore interior
THRESHOLD = "#8A8A8A"    # viability / throughput threshold lines: grey, never a data colour

# Finite element edges (Fig. 2B).
MESH_C = "#0072B2"

# Damage components: a third categorical axis.
COMP_C = {"shear": "#C7E9C0", "extensional": "#CB181D", "total": ANNOT}

# Published comparison data (Ning et al. 2018) is always open black squares.
NING_MARKER = dict(marker="s", markerfacecolor="none", markeredgecolor="#1A1A1A",
                   markeredgewidth=1.6, linestyle="none", markersize=10)

# Type.
PAGE_W_MM = 170.0                  # the journal column every manuscript figure is placed at
PAGE_W_IN = PAGE_W_MM / 25.4

PAPER = dict(title=10.0, label=9.0, tick=8.0, annot=8.0, legend=8.0,
             lw_data=2.0, lw_axis=1.1, ms=6.5, mew=0.9, dpi=600)

SLIDE = dict(title=16.0, label=13.5, tick=12.0, annot=12.0, legend=11.5,
             lw_data=2.6, lw_axis=1.2, ms=7.5, mew=1.0, dpi=200)


# One spelling per quantity, across all ten figures.
LBL = {
    "pressure":   "Applied pressure (kPa)",
    "diameter":   "Nozzle diameter (µm)",
    "entry_r":    "Nozzle entry radius (µm)",
    "axial_z":    "Axial position  $z$  (mm)",
    "radius_mm":  "Radius  $r$  (mm)",
    "radius_um":  "Radius  $r$  (µm)",
    "flow_q":     "Flow rate  $Q$  (µL/s)",   # the text and every key write µL/s
    "damage_t":   "Total damage  $D_t$  (%)",
    "damage_cum": "Cumulative  $D_t$  (%)",
}

# Panel labels: a bold capital, on its own, outside the axes.
PANEL_LABEL_PT = 12.0


def panel_label(ax, letter, dx=-34, dy=10, size=None, color=None):
    """Put `letter` at the panel's top-left corner, offset OUTSIDE the axes.

    Offsets are in points from the axes' top-left, so the label sits clear of the y-axis
    label and tick numbers whatever the data limits are, and it never moves when the axis
    autoscales. Returns the artist so a caller can measure it.

    One size for every figure in the manuscript (see PANEL_LABEL_PT above). `size` is for an
    artifact that is not one of the paper's figures.
    """
    a = ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=PANEL_LABEL_PT if size is None else size, fontweight="bold",
                    color=color if color is not None else INK,
                    ha="left", va="bottom", annotation_clip=False)
    a.set_gid("panel_label")     # so snap_panel_labels() can find it after the layout
    return a


def _live_ticklabels(ax, rend, which):
    """The tick labels actually on the axis, with the off-range ones dropped.

    Matplotlib keeps a Text artist for every tick the locator produced, including ticks
    outside the current limits: a log axis showing 10^1 to 10^4 still carries live artists
    for 10^5 and 10^6, positioned where they would be. They are not drawn, and
    `get_visible()` is True for them, so anything that measures tick labels naively picks up
    a box beyond the end of the panel, and a panel letter placed from it lands off the page.
    """
    ab = ax.get_window_extent(rend)
    out = []
    for t in (ax.get_xticklabels() if which == "x" else ax.get_yticklabels()):
        if not t.get_text().strip() or not t.get_visible():
            continue
        b = t.get_window_extent(rend)
        c = 0.5 * (b.x0 + b.x1) if which == "x" else 0.5 * (b.y0 + b.y1)
        lo, hi = (ab.x0, ab.x1) if which == "x" else (ab.y0, ab.y1)
        if lo - 2.0 <= c <= hi + 2.0:
            out.append((t.get_text(), b))
    return out


def _panel_box(ax, rend):
    """The panel's real extent: axes, both axis labels, tick labels and any key.

    NOT `ax.get_tightbbox()`. That method deliberately under-reports an axis label along its
    own axis (matplotlib shrinks it "for layout only" so a long label cannot blow up a
    constrained layout), and the difference is large enough that a panel letter placed from
    it can print across the top of a rotated y-axis label.
    """
    from matplotlib.transforms import Bbox
    boxes = [ax.get_window_extent(rend)]
    for art in (ax.xaxis.label, ax.yaxis.label):
        if art.get_text().strip():
            boxes.append(art.get_window_extent(rend))
    for w in ("x", "y"):
        boxes.extend(b for _, b in _live_ticklabels(ax, rend, w))
    leg = ax.get_legend()
    if leg is not None and leg.get_visible():
        boxes.append(leg.get_window_extent(rend))
    return Bbox.union(boxes)


# No border around the whole canvas. Each panel carries its own full box via
# rc()["axes.spines.*"], which is the frame a figure needs.


def snap_panel_labels(fig, dy_pt=4.0):
    """Align every panel letter with its own panel's leftmost ink. Call after layout.

    `panel_label` places the letter dx=-34 pt from the axes,
    a distance picked once to clear a y-axis label. What actually sits to the left of an axes
    varies: a long rotated label and five-character tick numbers on one panel, a short label
    and one digit on another. So the same -34 pt lands flush against the label on one panel and
    leaves the letter floating in the gap on the next.

    THE GAP IS 4 pt. Closer than that, a 12 pt letter sits directly on top of the y-axis label
    and the two read as one block; the air under the letter grows with the letter.

    THE LETTER GOES ABOVE THE WHOLE PANEL, not above the axes. Holding it at a fixed offset
    from the axes top puts it level with a y-axis label that is taller than its own axes, where
    it prints across the label. The panel's tight box already knows how far up its tallest ink
    reaches. This costs top margin, so a figure calling it must leave some
    (`tight_layout(rect=(0, 0, 1, 0.94))` or equivalent), and `assert_labels_clear` reports it
    if it does not.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    found = []
    for ax in fig.axes:
        for art in ax.texts:
            if art.get_gid() != "panel_label":
                continue
            # _panel_box never includes the letter itself, so this is idempotent: calling
            # it twice cannot walk the letter further out.
            found.append((ax, art, _panel_box(ax, rend)))

    # One height per row.
    rows: dict[float, float] = {}
    for ax, _, bb in found:
        key = round(ax.get_position().y1, 2)
        rows[key] = max(rows.get(key, 0.0), bb.y1)

    # The letter stays an offset from its own axes, in points, exactly as `panel_label` created it;
    # only the two numbers change.
    for ax, art, bb in found:
        top = rows[round(ax.get_position().y1, 2)]
        ab = ax.get_window_extent(rend)
        pt = 72.0 / fig.dpi
        art.xyann = ((bb.x0 - ab.x0) * pt, (top - ab.y1) * pt + dy_pt)


def keys_over_data(fig, pad=1.0, samples=600):
    """Every place a legend sits on top of the data it is supposed to explain.

    `assert_labels_clear` tests text against text and everything against the canvas edge.
    This tests a key against the CURVES UNDERNEATH IT, so a legend placed in what looked like
    empty space is still caught if the data later grows into that space.

    The line is SAMPLED, not tested at its vertices. Eight pressure points per curve leave
    segments long enough to pass straight through a key without putting a vertex inside it,
    and a vertex test would call that clear.

    Returns a list of (axes index, label, points covered) rather than raising, so a caller
    can report all of them at once.
    """
    from matplotlib.contour import ContourSet
    from matplotlib.path import Path
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hits = []
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        lb = leg.get_window_extent(rend).expanded(1.0, 1.0)
        lb = lb.padded(-pad) if hasattr(lb, "padded") else lb
        for line in ax.lines:
            if not line.get_visible():
                continue
            xy = line.get_xydata()
            if xy is None or len(xy) < 1:
                continue
            pts = ax.transData.transform(xy)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) > 1:
                # Resample the polyline so a long segment crossing the key is caught
                d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
                if d[-1] > 0:
                    t = np.linspace(0, d[-1], samples)
                    pts = np.column_stack([np.interp(t, d, pts[:, 0]),
                                           np.interp(t, d, pts[:, 1])])
            n = int(((pts[:, 0] >= lb.x0) & (pts[:, 0] <= lb.x1) &
                     (pts[:, 1] >= lb.y0) & (pts[:, 1] <= lb.y1)).sum())
            if n:
                hits.append((i, line.get_label(), n))
        for coll in ax.collections:
            # A contour set is tested by its paths.
            if isinstance(coll, ContourSet):
                verts = []
                for path in coll.get_paths():
                    v = path.vertices
                    if path.codes is not None:
                        v = v[(path.codes != Path.CLOSEPOLY) & (path.codes != Path.STOP)]
                    if len(v):
                        verts.append(v)
                if not verts:
                    continue
                pts = coll.get_transform().transform(np.concatenate(verts))
            else:
                off = coll.get_offsets()
                if off is None or len(off) == 0:
                    continue
                pts = ax.transData.transform(np.asarray(off))
            pts = pts[np.isfinite(pts).all(axis=1)]
            n = int(((pts[:, 0] >= lb.x0) & (pts[:, 0] <= lb.x1) &
                     (pts[:, 1] >= lb.y0) & (pts[:, 1] <= lb.y1)).sum())
            if n:
                hits.append((i, coll.get_label() or "<collection>", n))
        # Bars.
        from matplotlib.patches import Rectangle
        for patch in ax.patches:
            if not (isinstance(patch, Rectangle) and patch.get_visible()):
                continue
            bb = patch.get_window_extent(rend)
            ox = min(bb.x1, lb.x1) - max(bb.x0, lb.x0)
            oy = min(bb.y1, lb.y1) - max(bb.y0, lb.y0)
            if ox > 0 and oy > 0:
                hits.append((i, patch.get_label() or "<bar>", int(ox * oy)))
    return hits


def assert_keys_clear_of_data(fig):
    """Raise if any legend covers the data. Called for every figure from `save_figure`,
    so a generator cannot forget it and a later layout change cannot quietly reintroduce it."""
    hits = keys_over_data(fig)
    if hits:
        lines = "\n".join(f"    axes {i}: key covers {lab!r} at {n} sampled points"
                           for i, lab, n in hits)
        raise AssertionError("a legend is drawn on top of the data it explains:\n" + lines)


INK_MARGIN_MM = 0.4   # the least clear canvas allowed between any ink and the page edge


def ink_margins_mm(fig):
    """Clear canvas on each side, in mm, measured on the RENDERED PIXELS: the first and last
    painted row and column. Layout boxes cannot be used, because an axes laid over the whole
    canvas (a schematic) reports the whole canvas as its extent even with its axis off, and
    text cut by the canvas edge still reports its full box."""
    import numpy as np
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = (rgba[..., 3] > 10) & (rgba[..., :3] < 245).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if not len(rows):
        return None
    px = 25.4 / fig.dpi
    h, w = ink.shape
    return {"left": cols[0] * px, "right": (w - 1 - cols[-1]) * px,
            "top": rows[0] * px, "bottom": (h - 1 - rows[-1]) * px}


def assert_ink_inside_canvas(fig, min_mm=INK_MARGIN_MM):
    """Raise if any painted pixel lies within `min_mm` of the canvas edge.

    Called for every figure from `save_figure` (unless the save re-crops with a tight bbox).
    Panel letters are placed by `snap_panel_labels` AFTER the layout is solved, so a layout that
    left no headroom above the top panels puts the letters partly off the page, and nothing in
    the layout itself notices: Figures 4 and 7 shipped with A and B cut at the top edge."""
    m = ink_margins_mm(fig)
    bad = {side: v for side, v in (m or {}).items() if v < min_mm}
    if bad:
        raise AssertionError("ink at or past the canvas edge: " + ", ".join(
            f"{side} {v:.1f} mm" for side, v in bad.items()) + f" (need {min_mm} mm)")
    # A text wholly off the canvas paints no pixel, so the pixels cannot see it; its own box can.
    from matplotlib.text import Annotation, Text
    rend, fb = fig.canvas.get_renderer(), fig.bbox
    ticks = {lab for a in fig.axes for axis in (a.xaxis, a.yaxis)
             for tk in axis.get_major_ticks() + axis.get_minor_ticks()
             for lab in (tk.label1, tk.label2)}
    for t in fig.findobj(Text):
        if t in ticks or not (t.get_visible() and t.get_text().strip()):
            continue
        if isinstance(t, Annotation):
            if not t._check_xy(rend):
                continue                 # clipped by annotation_clip: not drawn
        elif t.get_clip_on() and t.axes is not None and \
                not t.axes.bbox.overlaps(t.get_window_extent(rend)):
            continue                     # clipped away by its own axes: not drawn
        bb = t.get_window_extent(rend)
        if bb.x1 < fb.x0 or bb.x0 > fb.x1 or bb.y1 < fb.y0 or bb.y0 > fb.y1:
            raise AssertionError(f"text {t.get_text()!r} lies wholly off the canvas")


def assert_labels_clear(fig, axes, slack=1.0, min_gap_frac=0.25):
    """Fail if any tick label overlaps its neighbour, or any key leaves its own panel.

    Three panels across the journal's 170 mm width leave a 36 mm plot box,
    and every text decision in one is a near-miss: a key can cross into the neighbouring
    panel's y-axis label, and adjacent diameter labels can merge into one string such as
    "O100O200O250O330". A check that tests plotted samples against a key sees neither, so this
    one tests text against text.

    `scripts/tests/test_labels_clear.py` builds a panel with deliberately colliding tick labels
    and requires this check to fail on it.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        for which in ("x", "y"):
            # Sorted by position, not by tick order.
            boxes = sorted(_live_ticklabels(ax, rend, which),
                           key=lambda tb: tb[1].x0 if which == "x" else tb[1].y0)
            # A gap, not merely an absence of overlap.
            floor = min_gap_frac * max((b.height for _, b in boxes), default=0.0)
            for (t1, b1), (t2, b2) in zip(boxes, boxes[1:]):
                gap = (b2.x0 - b1.x1) if which == "x" else (b2.y0 - b1.y1)
                if gap < floor - slack:
                    bad.append(f"axes {i}: {which} tick labels {t1!r} and {t2!r} are "
                               f"{gap:.1f} px apart, floor {floor:.1f} px")
        leg = ax.get_legend()
        if leg is not None:
            lb, ab = leg.get_window_extent(rend), ax.get_window_extent(rend)
            if lb.x0 < ab.x0 - slack or lb.x1 > ab.x1 + slack:
                bad.append(f"axes {i}: key spans x {lb.x0:.0f}-{lb.x1:.0f}, "
                           f"panel {ab.x0:.0f}-{ab.x1:.0f}")
    # And nothing may leave the canvas.
    fb = fig.bbox
    for ax in np.atleast_1d(axes).ravel():
        for art in [ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            if not art.get_text().strip() or not art.get_visible():
                continue
            b = art.get_window_extent(rend)
            if (b.x0 < fb.x0 - slack or b.x1 > fb.x1 + slack
                    or b.y0 < fb.y0 - slack or b.y1 > fb.y1 + slack):
                bad.append(f"{art.get_text()[:34]!r} runs off the canvas: "
                           f"x {b.x0:.0f}-{b.x1:.0f}, y {b.y0:.0f}-{b.y1:.0f} "
                           f"on {fb.x1:.0f} x {fb.y1:.0f}")
    assert not bad, "label collisions:\n  " + "\n  ".join(bad)

# The house look: a full box rather than two spines, a visible grid under the data, bold centred panel
# titles, bold axis labels, markers big enough to read with a white edge that separates them from the
# line they sit on, and legends inside a frame instead of floating.
GRID_C = "#D9D9D9"
LEGEND_EDGE = "#9A9A9A"


def rc(profile: dict, family: str = "sans") -> dict:
    """matplotlib rcParams for one profile."""
    fonts = (["DejaVu Sans"] if family == "sans" else ["DejaVu Serif"])
    return {
        "font.family": "sans-serif" if family == "sans" else "serif",
        "font.sans-serif": fonts if family == "sans" else ["DejaVu Sans"],
        "font.serif": fonts if family != "sans" else ["DejaVu Serif"],

        "axes.titlesize": profile["title"],
        "axes.titleweight": "bold",
        "axes.titlepad": 9.0,
        "axes.labelsize": profile["label"],
        "axes.labelweight": "bold",
        "xtick.labelsize": profile["tick"],
        "ytick.labelsize": profile["tick"],

        # Full box, not two spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": profile["lw_axis"],
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid under the data
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_C,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,

        # Framed legend
        "legend.fontsize": profile["legend"],
        "legend.frameon": True,
        # Opaque. With any transparency, a curve rising behind the key prints as a pale ghost
        # line inside it, which reads as an edited image rather than a plot.
        "legend.framealpha": 1.0,
        "legend.edgecolor": LEGEND_EDGE,
        "legend.fancybox": False,
        "legend.borderpad": 0.5,

        "lines.linewidth": profile["lw_data"],
        "lines.markersize": profile["ms"],
        "lines.markeredgewidth": profile["mew"],
        "lines.markeredgecolor": "white",

        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def dia_color(d) -> str:
    """Colour for a tip diameter in µm. Raises rather than inventing one, so a new needle
    is a deliberate palette decision instead of a silent matplotlib default."""
    key = int(round(float(d)))
    if key not in DIA_C:
        raise KeyError(f"no canonical colour for Ø{key} µm; add it to DIA_C in l8_style.py "
                       f"rather than letting matplotlib choose")
    return DIA_C[key]


def dia_marker(d) -> str:
    """Marker for a tip diameter in µm. Raises rather than defaulting, for the same reason
    `dia_color` does: a new needle is a deliberate decision, not a silent matplotlib pick."""
    key = int(round(float(d)))
    if key not in DIA_M:
        raise KeyError(f"no canonical marker for Ø{key} µm; add it to DIA_M in l8_style.py")
    return DIA_M[key]


def dia_label(d) -> str:
    """One spelling of a diameter label, everywhere."""
    return f"Ø{int(round(float(d)))} µm"
