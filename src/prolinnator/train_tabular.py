from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from prolinnator.dataset import load_dataset
from prolinnator.train import _split_dataset


BLUE_THEME = {
    "bg": "#f3f8ff",
    "panel": "#eaf2ff",
    "grid": "#c5d9f2",
    "text": "#0e3a66",
    "line_1": "#1f6fd5",
    "line_2": "#2ea3b7",
    "accent": "#7fc8f8",
    "cm_low": "#eff7ff",
    "cm_mid": "#7ab8ff",
    "cm_high": "#0b4ea2",
}


def _apply_custom_plot_style() -> None:
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
            "legend.frameon": True,
            "legend.facecolor": "#f6fbff",
            "legend.edgecolor": BLUE_THEME["grid"],
            "axes.grid": True,
            "grid.alpha": 0.5,
        }
    )


_apply_custom_plot_style()


def _extract_labeled_arrays(dataset: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[str] = []

    for data in dataset:
        mask = data.label_mask.cpu().numpy().astype(bool)
        if not np.any(mask):
            continue
        x = data.x.cpu().numpy()[mask]
        y = data.y.cpu().numpy()[mask].astype(int)
        xs.append(x)
        ys.append(y)
        groups.extend([str(data.protein_id)] * int(mask.sum()))

    if not xs:
        raise ValueError("No labeled residues were extracted from the dataset split")

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.array(groups)


def _fit_model(name: str, args: argparse.Namespace):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )

    if name == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=args.svm_c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(args.mlp_hidden_dim, args.mlp_hidden_dim // 2),
                        activation="relu",
                        alpha=args.mlp_alpha,
                        batch_size=args.mlp_batch_size,
                        learning_rate_init=args.mlp_lr,
                        max_iter=args.mlp_max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=args.gbm_estimators,
            max_depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
            subsample=args.gbm_subsample,
            colsample_bytree=args.gbm_colsample_bytree,
            reg_lambda=args.gbm_reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=args.gbm_estimators,
            max_depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
            subsample=args.gbm_subsample,
            colsample_bytree=args.gbm_colsample_bytree,
            reg_lambda=args.gbm_reg_lambda,
            objective="binary",
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=args.gbm_estimators,
            depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
            l2_leaf_reg=args.gbm_reg_lambda,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.seed,
            verbose=False,
            thread_count=-1,
        )

    raise ValueError(f"Unsupported model: {name}")


def _predict_scores(model, name: str, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if name == "linear_svm":
        scores = model.decision_function(x)
        preds = (scores >= 0.0).astype(int)
        return scores, None, preds

    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return probs, probs, preds


def _compute_metrics(y_true: np.ndarray, score: np.ndarray, probs: np.ndarray | None, preds: np.ndarray) -> Dict[str, float | int | None]:
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel().tolist()
    metrics: Dict[str, float | int | None] = {
        "n": int(y_true.size),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "brier": None,
    }
    if probs is not None:
        metrics["brier"] = float(brier_score_loss(y_true, probs))
    return metrics


def _plot_pr_curve(y_true: np.ndarray, score: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.plot(recall, precision, linewidth=2.4, color=BLUE_THEME["line_1"])
        ax.fill_between(recall, precision, alpha=0.15, color=BLUE_THEME["accent"])
    else:
        ax.text(0.1, 0.5, "Insufficient class diversity for PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_log_roc(y_true: np.ndarray, score: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, score)
        fpr = np.clip(fpr, 1e-6, 1.0)
        ax.semilogx(fpr, tpr, linewidth=2.4, color=BLUE_THEME["line_2"])
        ax.semilogx([1e-4, 1.0], [1e-4, 1.0], linestyle="--", linewidth=1.2, color=BLUE_THEME["grid"])
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.text(0.1, 0.5, "Insufficient class diversity for ROC curve")
    ax.set_xlabel("False Positive Rate (log scale)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, preds: np.ndarray, out_path: Path, title: str) -> None:
    cm_counts = confusion_matrix(y_true.astype(int), preds.astype(int), labels=[0, 1]) if y_true.size > 0 else np.zeros((2, 2), dtype=int)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "prolinnator_blues", [BLUE_THEME["cm_low"], BLUE_THEME["cm_mid"], BLUE_THEME["cm_high"]]
    )
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(f"{title} (Normalized)")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n(n={cm_counts[i, j]})",
                ha="center",
                va="center",
                color=("white" if cm_norm[i, j] > 0.55 else BLUE_THEME["text"]),
                fontsize=10,
                fontweight="semibold",
            )
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _extract_loss_history(model: Any, name: str) -> Dict[str, List[float]] | None:
    if name == "mlp":
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "loss_curve_") and len(getattr(clf, "loss_curve_", [])) > 0:
            return {"train_loss": [float(v) for v in clf.loss_curve_]}
        return None

    if name == "catboost":
        try:
            evals = model.get_evals_result()
        except Exception:
            return None
        hist: Dict[str, List[float]] = {}
        if "learn" in evals and "Logloss" in evals["learn"]:
            hist["train_loss"] = [float(v) for v in evals["learn"]["Logloss"]]
        if "validation" in evals and "Logloss" in evals["validation"]:
            hist["val_loss"] = [float(v) for v in evals["validation"]["Logloss"]]
        return hist if hist else None

    if name == "xgboost" and hasattr(model, "evals_result"):
        try:
            evals = model.evals_result()
        except Exception:
            return None
        hist = {}
        if "validation_0" in evals and "logloss" in evals["validation_0"]:
            hist["train_loss"] = [float(v) for v in evals["validation_0"]["logloss"]]
        if "validation_1" in evals and "logloss" in evals["validation_1"]:
            hist["val_loss"] = [float(v) for v in evals["validation_1"]["logloss"]]
        return hist if hist else None

    if name == "lightgbm" and hasattr(model, "evals_result_"):
        evals = getattr(model, "evals_result_", {})
        hist = {}
        if "training" in evals and "binary_logloss" in evals["training"]:
            hist["train_loss"] = [float(v) for v in evals["training"]["binary_logloss"]]
        if "valid_1" in evals and "binary_logloss" in evals["valid_1"]:
            hist["val_loss"] = [float(v) for v in evals["valid_1"]["binary_logloss"]]
        return hist if hist else None

    return None


def _plot_loss_history(history: Dict[str, List[float]] | None, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if history is None or "train_loss" not in history:
        ax.text(0.15, 0.52, "Loss history is not available for this model.")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train loss", linewidth=2.4, color=BLUE_THEME["line_1"])
    if "val_loss" in history and len(history["val_loss"]) == len(history["train_loss"]):
        ax.plot(epochs, history["val_loss"], label="val loss", linewidth=2.4, color=BLUE_THEME["line_2"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_tabular(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        csv_path=Path(args.csv),
        structures_dir=Path(args.structures_dir),
        ddg_threshold=args.ddg_threshold,
        distance_cutoff=args.distance_cutoff,
        single_pdb=args.single_pdb,
        single_chain=args.single_chain,
        embeddings_dir=(Path(args.embeddings_dir) if args.embeddings_dir else None),
        embedding_strict=args.embedding_strict,
    )

    train_set, val_set, test_set = _split_dataset(
        dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    eval_set = val_set if len(test_set) == 0 else test_set
    eval_name = "val" if len(test_set) == 0 else "test"
    print(f"split_sizes train={len(train_set)} val={len(val_set)} test={len(test_set)} eval={eval_name}")

    x_train, y_train, _ = _extract_labeled_arrays(train_set)
    x_eval, y_eval, _ = _extract_labeled_arrays(eval_set)
    print(f"labeled_residues train={x_train.shape[0]} eval={x_eval.shape[0]} feature_dim={x_train.shape[1]}")

    summary: Dict[str, Dict[str, float | int | None] | Dict[str, str]] = {}
    for model_name in args.models:
        print(f"training_model name={model_name}")
        try:
            model = _fit_model(model_name, args)
        except Exception as exc:
            msg = f"model_unavailable name={model_name} reason={exc}"
            print(msg)
            summary[model_name] = {"status": "skipped", "reason": str(exc)}
            continue

        fit_kwargs: Dict[str, Any] = {}
        if model_name in {"catboost", "xgboost", "lightgbm"}:
            fit_kwargs["eval_set"] = [(x_train, y_train), (x_eval, y_eval)]

        model.fit(x_train, y_train, **fit_kwargs)
        scores, probs, preds = _predict_scores(model, model_name, x_eval)
        metrics = _compute_metrics(y_eval, scores, probs, preds)
        summary[model_name] = metrics
        loss_history = _extract_loss_history(model, model_name)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = model_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{eval_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if loss_history is not None:
            with open(model_dir / "loss_history.json", "w", encoding="utf-8") as f:
                json.dump(loss_history, f, indent=2)

        _plot_pr_curve(y_eval, scores, plot_dir / f"{eval_name}_pr_curve.png", title=f"PR Curve ({model_name})")
        _plot_log_roc(y_eval, scores, plot_dir / f"{eval_name}_log_roc_curve.png", title=f"Log-ROC ({model_name})")
        _plot_confusion_matrix(
            y_eval,
            preds,
            plot_dir / f"{eval_name}_confusion_matrix.png",
            title=f"Confusion Matrix ({model_name})",
        )
        _plot_loss_history(loss_history, plot_dir / f"{eval_name}_loss_history.png", title=f"Loss History ({model_name})")

        print(
            f"result model={model_name} "
            f"pr_auc={float(metrics['pr_auc']):.4f} "
            f"roc_auc={float(metrics['roc_auc']):.4f} "
            f"brier={metrics['brier']}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved tabular baseline summary to: {output_dir / 'summary.json'}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train tabular residue-level baselines")
    p.add_argument("--csv", required=True, help="Path to mutation CSV")
    p.add_argument("--structures-dir", required=True, help="Directory containing PDB files")
    p.add_argument("--single-pdb", default=None, help="Optional single-protein mode PDB filename")
    p.add_argument("--single-chain", default="A", help="Chain ID for single-protein mode")
    p.add_argument("--ddg-threshold", type=float, default=1.0)
    p.add_argument("--distance-cutoff", type=float, default=8.0)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "linear_svm", "mlp", "xgboost", "lightgbm", "catboost"],
        default=["random_forest", "linear_svm", "mlp", "xgboost", "lightgbm", "catboost"],
        help="Tabular baseline models to train.",
    )
    p.add_argument(
        "--embeddings-dir",
        default=None,
        help="Optional directory containing per-protein residue embeddings (.npy or .npz).",
    )
    p.add_argument(
        "--embedding-strict",
        action="store_true",
        help="Fail if an embedding file is missing or malformed when --embeddings-dir is provided.",
    )
    p.add_argument("--rf-estimators", type=int, default=400)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-leaf", type=int, default=2)
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--mlp-hidden-dim", type=int, default=128)
    p.add_argument("--mlp-alpha", type=float, default=1e-4)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch-size", type=int, default=256)
    p.add_argument("--mlp-max-iter", type=int, default=100)
    p.add_argument("--gbm-estimators", type=int, default=500)
    p.add_argument("--gbm-max-depth", type=int, default=6)
    p.add_argument("--gbm-learning-rate", type=float, default=0.05)
    p.add_argument("--gbm-subsample", type=float, default=0.8)
    p.add_argument("--gbm-colsample-bytree", type=float, default=0.8)
    p.add_argument("--gbm-reg-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/tabular_baselines")
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    train_tabular(args)