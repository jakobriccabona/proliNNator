from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE_THEME = {
    "bg": "#f3f8ff",
    "panel": "#eaf2ff",
    "grid": "#c5d9f2",
    "text": "#0e3a66",
    "cm_low": "#eff7ff",
    "cm_mid": "#7ab8ff",
    "cm_high": "#0b4ea2",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": BLUE_THEME["panel"],
            "figure.facecolor": BLUE_THEME["bg"],
            "axes.edgecolor": BLUE_THEME["grid"],
            "axes.labelcolor": BLUE_THEME["text"],
            "axes.titlecolor": BLUE_THEME["text"],
            "xtick.color": BLUE_THEME["text"],
            "ytick.color": BLUE_THEME["text"],
            "grid.color": BLUE_THEME["grid"],
            "text.color": BLUE_THEME["text"],
        }
    )


def cm_from_counts(d: dict) -> np.ndarray:
    return np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]], dtype=int)


def draw_cm(ax, counts: np.ndarray, title: str) -> None:
    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "prolinnator_blues", [BLUE_THEME["cm_low"], BLUE_THEME["cm_mid"], BLUE_THEME["cm_high"]]
    )
    im = ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n(n={counts[i, j]})",
                ha="center",
                va="center",
                color=("white" if norm[i, j] > 0.55 else BLUE_THEME["text"]),
                fontsize=10,
                fontweight="semibold",
            )
    return im


def plot_pair(gnn_counts: np.ndarray, cat_counts: np.ndarray, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    im = draw_cm(axes[0], gnn_counts, "GNN")
    draw_cm(axes[1], cat_counts, "CatBoost")
    fig.suptitle(suptitle, fontsize=13)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.03)
    cbar.set_label("Normalized frequency")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    apply_style()
    in_path = Path("outputs/threshold_sweep_omega_hbond_f1.json")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing sweep file: {in_path}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    out_dir = Path("outputs/threshold_sweep_omega_hbond_f1_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn_default = cm_from_counts(payload["gnn"]["test_at_0.5"])
    cat_default = cm_from_counts(payload["catboost"]["test_at_0.5"])
    plot_pair(
        gnn_default,
        cat_default,
        "Confusion Matrices at Threshold 0.50",
        out_dir / "confusion_default_0p5_side_by_side.png",
    )

    gnn_tuned = cm_from_counts(payload["gnn"]["test_at_tuned_threshold"])
    cat_tuned = cm_from_counts(payload["catboost"]["test_at_tuned_threshold"])
    gnn_thr = payload["gnn"]["test_at_tuned_threshold"]["threshold"]
    cat_thr = payload["catboost"]["test_at_tuned_threshold"]["threshold"]
    plot_pair(
        gnn_tuned,
        cat_tuned,
        f"Confusion Matrices at Tuned Thresholds (GNN={gnn_thr:.2f}, CatBoost={cat_thr:.2f})",
        out_dir / "confusion_tuned_side_by_side.png",
    )

    print(f"saved={out_dir / 'confusion_default_0p5_side_by_side.png'}")
    print(f"saved={out_dir / 'confusion_tuned_side_by_side.png'}")


if __name__ == "__main__":
    main()
