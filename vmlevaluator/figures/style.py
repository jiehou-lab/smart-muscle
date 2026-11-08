"""Figure styling: palettes, textures, typography.

The bar figures and the ROC figure encode different things, so they use
different palettes. This is deliberate and matches the submitted figures.

Bar figures (1 and 3) — colour encodes **input**, not model
--------------------------------------------------------
Each panel shows three model families side by side, and within each family a
pair of bars: image-only and image-plus-morphometrics. The family is already
named on the x-axis, so spending colour on it would be redundant. Colour
instead carries the comparison the figure exists to make — what happens when
the two slide-level morphometrics are added — and the legend therefore has
exactly two entries. Hatching repeats that same distinction, so the figure
survives grayscale printing and colour-blind viewing.

Figure 1 and Figure 3 use different pairs so the patch-level and slide-level
figures are not mistaken for one another at a glance::

    figure 1 (patch)   image #2E5A88 (steel blue)   fusion #2F6B4F (dark green)
    figure 3 (WSI)     image #C9A66B (tan)          fusion #5C4A6E (dark purple)

ROC figure (2) — colour encodes family
--------------------------------------
A ROC panel has no x-axis categories, so family has to be carried by the line
itself: colour plus a distinct line style and marker.

Accessibility
-------------
``node scripts/validate_palette.js`` (the ``dataviz`` six checks) on the two
bar pairs, against a white surface:

===================  ==================================================
``#2E5A88,#2F6B4F``  FAILs the chroma floor (0.090, 0.078) and the
                     normal-vision floor (ΔE 12.1 < 15); passes CVD
                     separation (ΔE 11.4 deuteranopia).
``#C9A66B,#5C4A6E``  FAILs the chroma floor; passes CVD separation
                     (ΔE 31.9) and the normal-vision floor (ΔE 33.1).
                     The tan warns on contrast (2.24:1).
===================  ==================================================

These are the colours of record, so they ship as the default. The failures are
answered rather than ignored: every bar carries a hatch, a black edge and a
printed value, so identity and magnitude are both recoverable without colour.
For projection, posters and talks, switch to the validated alternative, which
passes all six checks (normal-vision ΔE 26.4, protanopia ΔE 21.1)::

    from vmlevaluator.figures import style
    style.use_palette("accessible")
"""

from __future__ import annotations

from typing import Dict, Sequence

__all__ = [
    "PALETTES",
    "FUSION_PALETTES",
    "ACTIVE",
    "use_palette",
    "family_color",
    "fusion_colors",
    "fusion_color",
    "fusion_hatch",
    "roc_style",
    "apply_rcparams",
    "FIGURE_SIZES",
    "label_bars",
    "format_axis",
]

# ── bar figures: colour by input ─────────────────────────────────────────────

#: ``{palette name: {figure key: {"image": hex, "fusion": hex}}}``. The figure
#: keys are ``patch`` (Figure 1), ``wsi`` (Figure 3) and ``transfer``
#: (Figure 4, which has a single family and so reuses the patch pair).
FUSION_PALETTES: Dict[str, Dict[str, Dict[str, str]]] = {
    "manuscript": {
        "patch": {"image": "#2E5A88", "fusion": "#2F6B4F"},
        "wsi": {"image": "#C9A66B", "fusion": "#5C4A6E"},
        "transfer": {"image": "#2E5A88", "fusion": "#2F6B4F"},
    },
    # Validated: all six checks pass against a white surface.
    "accessible": {
        "patch": {"image": "#3D6DB5", "fusion": "#C2622A"},
        "wsi": {"image": "#3D6DB5", "fusion": "#C2622A"},
        "transfer": {"image": "#3D6DB5", "fusion": "#C2622A"},
    },
}

#: Hatch per input. Solid image-only, diagonal for fusion — the same
#: distinction the colour makes, in a channel that survives a photocopier.
FUSION_HATCH: Dict[str, str] = {"image": "", "fusion": "///"}

# ── ROC figure: colour by family ─────────────────────────────────────────────

#: ``XGB`` carries the same slot as ``RF``: they are the middle family in the
#: two versions of this study, never both at once, so giving them one colour
#: keeps a regenerated figure comparable with the originally submitted one and
#: lets this file be identical in both packages.
PALETTES: Dict[str, Dict[str, str]] = {
    "manuscript": {"LR": "#4C72B0", "RF": "#DD8452", "XGB": "#DD8452",
                   "ResNet50": "#55A868"},
    "accessible": {"LR": "#3D6DB5", "RF": "#D9822B", "XGB": "#D9822B",
                   "ResNet50": "#7A4FA3"},
}

#: Line style and marker per family, so ROC curves separate without colour.
ROC_STYLE: Dict[str, Dict] = {
    "LR": {"marker": "o"},
    "RF": {"marker": "s"},
    "XGB": {"marker": "s"},
    "ResNet50": {"marker": "^"},
}

ACTIVE = {
    "name": "manuscript",
    "colors": dict(PALETTES["manuscript"]),
    "fusion": {k: dict(v) for k, v in FUSION_PALETTES["manuscript"].items()},
}

FIGURE_SIZES = {
    "single": (5.2, 4.2),
    "wide": (9.0, 4.2),
    "triptych": (14.0, 4.8),
    "square": (5.2, 5.2),
}

TYPOGRAPHY = {
    "title": 13,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}

#: Bar panels share one y-range so panels can be compared by eye, and it
#: leaves room above a bar at 1.000 for its printed value.
BAR_YLIM = (0.0, 1.07)
BAR_YTICKS = [i / 10 for i in range(11)]


def use_palette(name: str) -> Dict[str, str]:
    """Switch the active palette (``manuscript`` or ``accessible``)."""
    if name not in PALETTES:
        raise ValueError(f"Unknown palette {name!r}; choose from {sorted(PALETTES)}")
    ACTIVE["name"] = name
    ACTIVE["colors"] = dict(PALETTES[name])
    ACTIVE["fusion"] = {k: dict(v) for k, v in FUSION_PALETTES[name].items()}
    return ACTIVE["colors"]


def family_color(family: str) -> str:
    return ACTIVE["colors"].get(family, "#7f7f7f")


def fusion_colors(figure_key: str = "patch") -> Dict[str, str]:
    """The ``{"image": hex, "fusion": hex}`` pair for one bar figure."""
    return ACTIVE["fusion"].get(figure_key, ACTIVE["fusion"]["patch"])


def fusion_color(figure_key: str, morphometrics: bool) -> str:
    return fusion_colors(figure_key)["fusion" if morphometrics else "image"]


def fusion_hatch(morphometrics: bool) -> str:
    return FUSION_HATCH["fusion" if morphometrics else "image"]


def roc_style(family: str, morphometrics: bool, *, n_points: int = 200) -> Dict:
    """Line properties for one ROC curve.

    Two independent non-colour channels carry the two distinctions: the
    **marker** identifies the model family and the **line style** says whether
    morphometrics were fused (solid = image only, dashed = fusion), mirroring
    solid-vs-hatched in the bar figures. Markers are thinned with
    ``markevery`` — a ROC curve over thousands of patches would otherwise
    render as a solid band of marker glyphs.
    """
    color = family_color(family)
    return {
        "color": color,
        "marker": ROC_STYLE.get(family, {}).get("marker", "o"),
        "linestyle": "--" if morphometrics else "-",
        "linewidth": 2.0,
        "markersize": 5,
        "markevery": max(1, int(n_points // 8)),
        "markerfacecolor": "white" if morphometrics else color,
        "markeredgecolor": color,
        "markeredgewidth": 1.2,
        "alpha": 0.95,
    }


def apply_rcparams() -> None:
    """Recessive axes, light grid, journal-ready fonts and 300 dpi export."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": TYPOGRAPHY["tick"],
            "axes.titlesize": TYPOGRAPHY["title"],
            "axes.labelsize": TYPOGRAPHY["label"],
            "xtick.labelsize": TYPOGRAPHY["tick"],
            "ytick.labelsize": TYPOGRAPHY["tick"],
            "legend.fontsize": TYPOGRAPHY["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "hatch.linewidth": 0.9,
            "hatch.color": "white",
        }
    )


def format_axis(ax, *, ylabel: str = "") -> None:
    """The shared y-axis treatment of the bar panels.

    Ticks every 0.10 with two decimals, a dotted rule at the ceiling, and a
    horizontal-only grid — vertical gridlines would cut through the grouped
    bars without helping anyone read a value.
    """
    ax.set_ylim(*BAR_YLIM)
    ax.set_yticks(BAR_YTICKS)
    ax.set_yticklabels([f"{v:.2f}" for v in BAR_YTICKS])
    ax.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0, zorder=0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)


def label_bars(ax, bars, values: Sequence[float], fmt: str = "{:.2f}") -> None:
    """Print each bar's value above it, in bold.

    Not decoration: the manuscript palette fails the normal-vision separation
    check on the Figure 1 pair, and the printed number is what guarantees a
    reader can always recover the value regardless of how the figure is
    reproduced.
    """
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=TYPOGRAPHY["annotation"],
            fontweight="bold",
            color="#222222",
        )
