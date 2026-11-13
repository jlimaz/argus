"""Plots for a run, only when matplotlib is importable (it is not on the laptop).

Returns the list of files written, or [] when matplotlib is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write(run: Path, metrics: dict[str, Any], calibration: dict[str, Any] | None) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = run / "plots"
    out_dir.mkdir(exist_ok=True)
    written = []
    classes = metrics["classes"]
    for split, r in metrics["splits"].items():
        if not r.get("window"):
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(r["window"]["confusion"], cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{split}: confusion (windows)")
        fig.tight_layout()
        path = out_dir / f"confusion_{split}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
        bins = [b for b in r["accuracy_vs_snr"]["bins"] if b["n"]]
        if bins:
            fig, ax = plt.subplots(figsize=(5, 3))
            mids = [((b["lo"] if b["lo"] is not None else b["hi"] - 5) + (b["hi"] if b["hi"] is not None else b["lo"] + 5)) / 2
                    for b in bins]
            ax.plot(mids, [b["accuracy"] for b in bins], marker="o")
            ax.set_xlabel("estimated SNR (dB)")
            ax.set_ylabel("accuracy")
            ax.set_ylim(0, 1)
            ax.set_title(f"{split}: accuracy vs SNR")
            fig.tight_layout()
            path = out_dir / f"snr_{split}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(str(path))
    if calibration:
        fig, ax = plt.subplots(figsize=(4, 4))
        for name in ("before", "after"):
            pts = [(b["confidence"], b["accuracy"]) for b in calibration["reliability"][name] if b["n"]]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title("reliability (val)")
        ax.legend()
        fig.tight_layout()
        path = out_dir / "reliability_val.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    return written
