"""Shared matplotlib style so every chart of the project looks the same."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE = "#2a78d6"      # main series (actual sales)
ORANGE = "#eb6834"    # second series (forecast)
AQUA = "#1baf7a"
GRAY = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 10,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def money(x, _=None):
    """Format a value in pounds: £1.2M, £350k."""
    if abs(x) >= 1e6:
        return f"£{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:.0f}k"
    return f"£{x:.0f}"


MONEY = FuncFormatter(money)
