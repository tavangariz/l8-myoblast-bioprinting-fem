"""Paths to the data.

Everything the scripts read ships under results/, so they run from a fresh clone with no
configuration. Optional environment variables:

    L8_DATA    a directory containing fem/exports/, to read data stored elsewhere
    L8_CKPT    the solver checkpoint (default results/fem/checkpoints/fem_solution.pkl)
    L8_FIGDIR  a second directory each figure is also written to

A missing input raises an error naming the path that was looked for.
"""
from __future__ import annotations

import os
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Root holding the solver exports. Repo-relative unless L8_DATA overrides it.
DATA = pathlib.Path(os.environ.get("L8_DATA", REPO / "results"))

#: Figures always land here, inside the repository.
FIGDIR = REPO / "results" / "figures"

#: Optional second directory each figure is also written to.
FIGMIRROR = pathlib.Path(os.environ["L8_FIGDIR"]) if os.environ.get("L8_FIGDIR") else None

#: The canonical 4 x 8 grid: four tip diameters, eight applied pressures, mesh M12,
#: solved on the cartridge-and-needle domain of Ning et al. (2018). The name carries the
#: date the grid was solved, so a figure always states which solution it was drawn from.
CANONICAL_GRID = DATA / "fem" / "exports" / "L8_FEM_canonical_2026-08-27.csv"


#: The converged FEM field the trajectory and contraction figures interpolate. It is a large
#: binary; L8_CKPT points at a copy elsewhere.
CHECKPOINT = pathlib.Path(
    os.environ.get("L8_CKPT", DATA / "fem" / "checkpoints" / "fem_solution.pkl"))

#: Wall shear along the capillary at the reference condition, Ø200 µm at 200 kPa.
WALL_PROFILE = DATA / "fem" / "exports" / "wall_shear_profile_D200_P200_M12.csv"


def require(path: pathlib.Path, what: str) -> pathlib.Path:
    """Return `path`, or explain precisely what is missing and how to point at it."""
    if not path.exists():
        raise SystemExit(
            f"missing {what}:\n    {path}\n\n"
            f"This file ships with the repository under results/. If your copy lives "
            f"elsewhere, set L8_DATA to the directory containing fem/exports/ and re-run:\n"
            f"    export L8_DATA=/path/to/data\n\n"
            f"For a checkpoint stored elsewhere, set L8_CKPT to its path."
        )
    return path


def flatten_png(path) -> None:
    """Composite a PNG onto white and store it as RGB.

    matplotlib writes RGBA whatever facecolor is set, and Word composites that alpha channel
    when it places the image, so a figure whose pixels are all white can still show a tinted
    background once placed. Some generators call fig.savefig directly rather than save_figure,
    so both paths call this one function.
    """
    import pathlib
    from PIL import Image
    path = pathlib.Path(path)
    if path.suffix.lower() != ".png":
        return
    with Image.open(path) as im:
        if im.mode == "RGB":
            return
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode in ("RGBA", "LA") else None)
        bg.save(path)


def save_figure(fig, name: str, dpi: int, tight: bool = False) -> list[pathlib.Path]:
    """Write `fig` to results/figures/<name>, and mirror it if L8_FIGDIR is set.

    `tight` passes bbox_inches="tight", which re-crops to the drawn content. It changes the
    output margins, so it is opt-in: a figure laid out with tight_layout alone must not
    silently acquire a different crop.

    Returns every path written, so the caller can report them.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    written = [FIGDIR / name]
    # Pad_inches is set explicitly rather than inherited: rcParams["savefig.pad_inches"] can be 0, and
    # with a tight bbox that puts the outermost glyph flush against the canvas edge and can clip a
    # panel letter.
    from l8_style import assert_ink_inside_canvas, assert_keys_clear_of_data
    assert_keys_clear_of_data(fig)
    if not tight:                       # a tight save re-crops, so its canvas is not this one
        assert_ink_inside_canvas(fig)
    fig.savefig(written[0], dpi=dpi, facecolor="white",
                **({"bbox_inches": "tight", "pad_inches": 0.06} if tight else {}))
    # Flatten every PNG to RGB at the writer, so the output mode is defined by construction (see
    # flatten_png).
    flatten_png(written[0])
    if FIGMIRROR is not None:
        FIGMIRROR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(written[0], FIGMIRROR / name)
        written.append(FIGMIRROR / name)
    return written
