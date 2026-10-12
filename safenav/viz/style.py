"""
style.py — one visual language for every SafeNav-RL figure.

Colours come from a colour-blind-checked categorical palette (fixed slot
order, never cycled) and a separate outcome palette; ink, grid and surface
colours are recessive so the data carries the contrast.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots (fixed order): one colour per method, everywhere.
METHOD_COLORS = {
    "PPO (filter in training)": "#2a78d6",
    "PPO (no filter in training)": "#eb6834",
    "PotentialField": "#1baf7a",
    "GoToGoal": "#eda100",
}
METHOD_ORDER = list(METHOD_COLORS)

# Outcome palette (status meaning; always shown with a legend / labels).
OUTCOME_COLORS = {"success": "#0ca30c", "timeout": "#eda100", "collision": "#d03b3b"}

OBSTACLE_FILL = "#d9d8d2"
OBSTACLE_EDGE = "#a3a29b"
MARGIN_EDGE = "#b8b7af"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "text.color": INK,
    })


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, MUTED)


def finish(fig, path, note: str | None = None, note_y: float = -0.02) -> None:
    """Optional footnote (data source), save and close."""
    if note:
        fig.text(0.01, note_y, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(path)
    plt.close(fig)
