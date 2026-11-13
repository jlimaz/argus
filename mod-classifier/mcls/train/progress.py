"""Loss curves for a run, redrawn after every epoch. No torch.

`write_plot(run_dir)` reads `history.jsonl` (one JSON row per epoch, written by
`mcls.train.loop`) and writes `progress.png`: train and validation loss on top,
validation macro-F1 below, with the best epoch marked, and keeps a copy of each
epoch's plot as `progress/epoch_NNN.png`. It needs matplotlib and
returns None without it. The training loop calls it inside a try/except, so a
plotting problem can never stop a run.

    python -m mcls.train.progress runs/<name>     # redraw by hand
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLOT_NAME = "progress.png"

# Colours: surface, ink and the first two categorical slots of the plot palette
# (blue, orange; checked for colour-blind separation and contrast on this surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
TRAIN = "#2a78d6"
VAL = "#eb6834"


def read_history(run_dir: Path) -> list[dict]:
    """Rows of history.jsonl in epoch order; a half-written last line is skipped."""
    path = Path(run_dir) / "history.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "epoch" in row:
            rows.append(row)
    return sorted(rows, key=lambda r: r["epoch"])


def summary(rows: list[dict]) -> str:
    """One line for the plot title and the email subject."""
    if not rows:
        return "no epochs yet"
    last = rows[-1]
    best = max(rows, key=lambda r: r["val_macro_f1"])
    return (f"epoch {last['epoch']}: train loss {last['train_loss']:.3f}, val loss {last['val_loss']:.3f}, "
            f"val macro-F1 {last['val_macro_f1']:.3f} (best {best['val_macro_f1']:.3f} at epoch {best['epoch']})")


def write_plot(run_dir: Path, total_epochs: int | None = None) -> Path | None:
    """Draw progress.png for the run. None when there is nothing to draw or no matplotlib."""
    run_dir = Path(run_dir)
    rows = read_history(run_dir)
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return None

    epochs = [r["epoch"] for r in rows]  # numbered from 0, as in the log and history.jsonl
    best = max(rows, key=lambda r: r["val_macro_f1"])
    right = max((total_epochs or 0) - 1, epochs[-1], 1)
    marker = "o" if len(rows) <= 25 else None

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28})
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)

    for key, colour, name in (("train_loss", TRAIN, "train"), ("val_loss", VAL, "validation")):
        values = [r[key] for r in rows]
        top.plot(epochs, values, color=colour, linewidth=2, marker=marker, markersize=4, label=name)
        top.annotate(f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=INK)
    top.set_title("Loss", loc="left", fontsize=10, color=INK, pad=6)
    top.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=2)

    f1 = [r["val_macro_f1"] for r in rows]
    bottom.plot(epochs, f1, color=VAL, linewidth=2, marker=marker, markersize=4)
    bottom.plot([best["epoch"]], [best["val_macro_f1"]], marker="o", markersize=9, color=VAL,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    bottom.annotate(f"best {best['val_macro_f1']:.3f}", (best["epoch"], best["val_macro_f1"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
    bottom.set_title("Validation macro-F1", loc="left", fontsize=10, color=INK, pad=6)
    bottom.set_ylim(0, 1.05)
    bottom.set_xlabel("epoch", fontsize=9, color=INK_SOFT)
    bottom.set_xlim(-0.5, right + 0.5)
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    fig.suptitle(f"{run_dir.name}\n{summary(rows)}", x=0.02, ha="left", fontsize=9, color=INK_SOFT)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.84, bottom=0.1)
    path = run_dir / PLOT_NAME
    tmp = run_dir / (PLOT_NAME + ".tmp")  # write then rename, so a reader never sees half a file
    fig.savefig(tmp, dpi=110, format="png", facecolor=SURFACE)
    plt.close(fig)
    tmp.replace(path)
    kept = run_dir / "progress" / f"epoch_{rows[-1]['epoch']:03d}.png"
    kept.parent.mkdir(exist_ok=True)
    kept.write_bytes(path.read_bytes())
    return path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m mcls.train.progress runs/<name>", file=sys.stderr)
        return 2
    rows = read_history(Path(args[0]))
    print(summary(rows))
    path = write_plot(Path(args[0]))
    if path is None and rows:
        print("matplotlib not importable: no plot written", file=sys.stderr)
    elif path:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
