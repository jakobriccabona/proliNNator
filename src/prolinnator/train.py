from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from prolinnator.dataset import load_dataset
from prolinnator.model import ProlineSiteGNN, masked_bce_loss, masked_focal_loss


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
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "black",
            "text.color": "black",
            "axes.linewidth": 2.0,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.framealpha": 0.95,
            "axes.grid": False,
        }
    )


_apply_custom_plot_style()


def _collect_targets_and_probs(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    probs = probs[mask].detach().cpu().numpy()
    ys = labels[mask].detach().cpu().numpy()
    return ys, probs


def _compute_train_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    pos_weight: float | None,
) -> torch.Tensor:
    if args.loss == "focal":
        return masked_focal_loss(
            logits,
            labels,
            mask,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            pos_weight=(pos_weight if args.use_pos_weight else None),
        )

    return masked_bce_loss(
        logits,
        labels,
        mask,
        pos_weight=(pos_weight if args.use_pos_weight else None),
    )


def _evaluate_loss(
    model: ProlineSiteGNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    pos_weight: float | None = None,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
            loss = _compute_train_loss(logits, batch.y, batch.label_mask, args=args, pos_weight=pos_weight)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate_reference_loss(model: ProlineSiteGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
            loss = masked_bce_loss(logits, batch.y, batch.label_mask, pos_weight=None)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _collect_predictions(model: ProlineSiteGNN, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys_all: List[np.ndarray] = []
    ps_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
            ys, ps = _collect_targets_and_probs(logits, batch.y, batch.label_mask)
            if ys.size > 0:
                ys_all.append(ys)
                ps_all.append(ps)

    if not ys_all:
        return np.array([]), np.array([])

    return np.concatenate(ys_all), np.concatenate(ps_all)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float | int]:
    if y_true.size == 0:
        return {
            "n": 0,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "n": int(y_true.size),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _plot_losses(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(history["epoch"], history["train_loss"], label="train loss", linewidth=2.4, color="#2c3e50")
    ax.plot(history["epoch"], history["val_loss"], label="val loss", linewidth=2.4, color="#e74c3c")
    ax.fill_between(history["epoch"], history["train_loss"], alpha=0.1, color="#2c3e50")
    ax.fill_between(history["epoch"], history["val_loss"], alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, linewidth=2.4, color="#2c3e50")
        ax.fill_between(recall, precision, alpha=0.15, color="#3498db")
    else:
        ax.text(0.1, 0.5, "Insufficient class diversity for PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_log_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fpr = np.clip(fpr, 1e-6, 1.0)
        ax.semilogx(fpr, tpr, linewidth=2.4, color="#2c3e50")
        ax.semilogx([1e-4, 1.0], [1e-4, 1.0], linestyle="--", linewidth=1.2, color="#999999")
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


def _plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int) if y_true.size > 0 else np.array([], dtype=int)
    cm_counts = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1]) if y_true.size > 0 else np.zeros((2, 2), dtype=int)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "prolinnator_grays", ["#ffffff", "#cccccc", "#000000"]
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
                color="black",
                fontsize=10,
                fontweight="semibold",
            )
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _split_dataset(dataset: list, train_fraction: float, val_fraction: float, test_fraction: float, seed: int):
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    if len(dataset) < 2:
        raise ValueError("Need at least 2 proteins/graphs for a split")

    groups = np.array([d.protein_id for d in dataset])
    indices = np.arange(len(dataset))

    # train/val only — no held-out test set
    if test_fraction == 0.0:
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
        idx_train, idx_val = next(splitter.split(indices, groups=groups))
        return [dataset[i] for i in idx_train], [dataset[i] for i in idx_val], []

    if len(dataset) < 3:
        raise ValueError("Need at least 3 proteins/graphs for train/val/test split")

    first = GroupShuffleSplit(n_splits=1, train_size=(train_fraction + val_fraction), random_state=seed)
    idx_train_val, idx_test = next(first.split(indices, groups=groups))

    train_rel = train_fraction / (train_fraction + val_fraction)
    second = GroupShuffleSplit(n_splits=1, train_size=train_rel, random_state=seed + 1)
    idx_train, idx_val = next(second.split(idx_train_val, groups=groups[idx_train_val]))

    train_idx = idx_train_val[idx_train]
    val_idx = idx_train_val[idx_val]

    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in idx_test]


def _to_ego_subgraph_samples(graphs: list, num_hops: int) -> list:
    samples = []
    aa_one_hot_dim = 20
    structural_prefix_dim = 27

    for graph in graphs:
        labeled_nodes = torch.nonzero(graph.label_mask, as_tuple=False).view(-1)
        if labeled_nodes.numel() == 0:
            continue

        for center in labeled_nodes.tolist():
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=int(center),
                num_hops=num_hops,
                edge_index=graph.edge_index,
                relabel_nodes=True,
                num_nodes=int(graph.x.shape[0]),
            )

            x_sub = graph.x[subset]
            base_feat_dim = int(x_sub.shape[1])
            y_sub = torch.zeros(subset.size(0), dtype=torch.float32)
            label_mask_sub = torch.zeros(subset.size(0), dtype=torch.bool)
            center_new_idx = int(mapping.item())
            y_sub[center_new_idx] = float(graph.y[int(center)].item())
            label_mask_sub[center_new_idx] = True

            # Mask sequence identity + sequence embeddings only for the target node.
            if x_sub.shape[1] >= aa_one_hot_dim:
                x_sub[center_new_idx, :aa_one_hot_dim] = 0.0
            if base_feat_dim > structural_prefix_dim:
                x_sub[center_new_idx, structural_prefix_dim:base_feat_dim] = 0.0

            # Add target-relative geometric and sequence-position context to every node.
            ca_dist = torch.zeros((subset.size(0), 1), dtype=torch.float32)
            if hasattr(graph, "ca_coords"):
                ca_sub = graph.ca_coords[subset]
                center_ca = graph.ca_coords[int(center)]
                ca_dist = torch.linalg.norm(ca_sub - center_ca, dim=1, keepdim=True)

            seq_sep = torch.zeros((subset.size(0), 1), dtype=torch.float32)
            if hasattr(graph, "residue_numbers"):
                center_resnum = graph.residue_numbers[int(center)]
                seq_sep = (graph.residue_numbers[subset] - center_resnum).abs().to(torch.float32).unsqueeze(1)

            x_sub = torch.cat([x_sub, ca_dist, seq_sep], dim=1)

            edge_attr_sub = None
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                edge_attr_sub = graph.edge_attr[edge_mask]

            sample = graph.__class__(
                x=x_sub,
                edge_index=sub_edge_index,
                edge_attr=edge_attr_sub,
                y=y_sub,
                label_mask=label_mask_sub,
            )

            sample.protein_id = graph.protein_id
            sample.chain_id = graph.chain_id
            if hasattr(graph, "residue_numbers"):
                sample.residue_numbers = graph.residue_numbers[subset]
                sample.center_residue_number = int(graph.residue_numbers[int(center)].item())
            if hasattr(graph, "ca_coords"):
                sample.ca_coords = graph.ca_coords[subset]
            sample.center_node_index = int(center)
            samples.append(sample)

    return samples


def _save_checkpoint(path: Path, model: ProlineSiteGNN, in_dim: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "edge_dim": getattr(model, "edge_dim", 0),
            "ddg_threshold": args.ddg_threshold,
            "distance_cutoff": args.distance_cutoff,
            "embeddings_dir": args.embeddings_dir,
        },
        path,
    )


def _load_model(path: Path, device: torch.device) -> ProlineSiteGNN:
    ckpt = torch.load(path, map_location=device)
    model = ProlineSiteGNN(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
        edge_dim=int(ckpt.get("edge_dim", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _load_pretrained_weights(model: ProlineSiteGNN, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(
        f"loaded_pretrained path={path} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )


def _set_encoder_trainable(model: ProlineSiteGNN, trainable: bool) -> None:
    for module in [model.convs, model.norms]:
        for p in module.parameters():
            p.requires_grad = trainable


def _build_optimizer(model: ProlineSiteGNN, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_params = list(model.convs.parameters()) + list(model.norms.parameters())
    head_params = list(model.head.parameters())

    if abs(args.encoder_lr_scale - 1.0) < 1e-12:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


def train(args: argparse.Namespace) -> None:
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    plot_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        csv_path=Path(args.csv),
        structures_dir=Path(args.structures_dir),
        ddg_threshold=args.ddg_threshold,
        distance_cutoff=args.distance_cutoff,
        single_pdb=args.single_pdb,
        single_chain=args.single_chain,
        embeddings_dir=(Path(args.embeddings_dir) if args.embeddings_dir else None),
        embedding_strict=args.embedding_strict,
        task=args.task,
        mask_sequence_features=args.mask_sequence_features,
    )

    train_set, val_set, test_set = _split_dataset(
        dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print(f"split_sizes_proteins train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    if args.sample_mode == "ego_subgraph":
        train_set = _to_ego_subgraph_samples(train_set, num_hops=args.subgraph_hops)
        val_set = _to_ego_subgraph_samples(val_set, num_hops=args.subgraph_hops)
        test_set = _to_ego_subgraph_samples(test_set, num_hops=args.subgraph_hops)
        print(
            f"sample_mode=ego_subgraph subgraph_hops={args.subgraph_hops} "
            f"split_sizes_samples train={len(train_set)} val={len(val_set)} test={len(test_set)}"
        )
    else:
        print(f"sample_mode=full_graph split_sizes_samples train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    no_test = len(test_set) == 0
    if no_test:
        print("mode=train_val_only test_eval=on_val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = val_loader if no_test else DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if len(train_set) == 0:
        raise ValueError("Training set is empty after preprocessing/sampling")
    in_dim = int(train_set[0].x.shape[1])
    edge_dim = int(train_set[0].edge_attr.shape[1]) if (hasattr(train_set[0], "edge_attr") and train_set[0].edge_attr is not None) else 0
    model = ProlineSiteGNN(in_dim=in_dim, hidden_dim=args.hidden_dim, layers=args.layers, dropout=args.dropout, edge_dim=edge_dim).to(device)
    if args.pretrained_checkpoint:
        _load_pretrained_weights(model, Path(args.pretrained_checkpoint), device)
    optimizer = _build_optimizer(model, args)

    encoder_frozen = bool(args.pretrained_checkpoint) and args.freeze_encoder_epochs > 0
    if encoder_frozen:
        _set_encoder_trainable(model, trainable=False)
        print(f"freeze_encoder enabled epochs=1..{args.freeze_encoder_epochs}")

    total_pos = 0
    total_neg = 0
    for d in train_set:
        y = d.y[d.label_mask]
        total_pos += int((y == 1).sum().item())
        total_neg += int((y == 0).sum().item())

    pos_weight = None
    if total_pos > 0 and total_neg > 0:
        pos_weight = float(total_neg / max(total_pos, 1))

    if pos_weight is not None:
        print(f"class_balance pos={total_pos} neg={total_neg} pos_weight={pos_weight:.4f}")
    else:
        print(f"class_balance pos={total_pos} neg={total_neg} pos_weight=NA")

    if args.loss == "focal":
        print(
            f"loss_config type=focal gamma={args.focal_gamma} "
            f"alpha={args.focal_alpha if args.focal_alpha is not None else 'None'} "
            f"use_pos_weight={args.use_pos_weight}"
        )
    else:
        print(f"loss_config type=bce use_pos_weight={args.use_pos_weight}")

    print(
        f"task={args.task} mask_sequence_features={args.mask_sequence_features} "
        f"encoder_lr_scale={args.encoder_lr_scale} pretrained_checkpoint={args.pretrained_checkpoint}"
    )

    best_val_loss = float("inf")
    best_val_pr = -1.0

    best_loss_path = ckpt_dir / "best_val_loss.pt"
    best_pr_path = ckpt_dir / "best_val_pr_auc.pt"
    early_stop_path = ckpt_dir / "early_stop_best.pt"
    last_path = ckpt_dir / "last_epoch.pt"

    if args.early_stopping_monitor == "val_pr_auc":
        best_monitor = -float("inf")
    else:
        best_monitor = float("inf")
    epochs_without_improvement = 0

    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if encoder_frozen and epoch == (args.freeze_encoder_epochs + 1):
            _set_encoder_trainable(model, trainable=True)
            encoder_frozen = False
            print(f"unfreeze_encoder epoch={epoch}")

        model.train()
        if encoder_frozen:
            for conv in model.convs:
                conv.eval()
            for norm in model.norms:
                norm.eval()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
            loss = _compute_train_loss(logits, batch.y, batch.label_mask, args=args, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = _evaluate_reference_loss(model, val_loader, device=device)
        y_val, p_val = _collect_predictions(model, val_loader, device)
        val_metrics = _compute_metrics(y_val, p_val)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": float(val_metrics["pr_auc"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_brier": float(val_metrics["brier"]),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(best_loss_path, model, in_dim, args)

        if not np.isnan(float(val_metrics["pr_auc"])) and float(val_metrics["pr_auc"]) > best_val_pr:
            best_val_pr = float(val_metrics["pr_auc"])
            _save_checkpoint(best_pr_path, model, in_dim, args)

        monitor_value = float(val_metrics["pr_auc"]) if args.early_stopping_monitor == "val_pr_auc" else float(val_loss)
        improved = False
        if args.early_stopping_monitor == "val_pr_auc":
            if not np.isnan(monitor_value) and (monitor_value - best_monitor) >= args.early_stopping_min_delta:
                improved = True
        else:
            if not np.isnan(monitor_value) and (best_monitor - monitor_value) >= args.early_stopping_min_delta:
                improved = True

        if improved:
            best_monitor = monitor_value
            epochs_without_improvement = 0
            _save_checkpoint(early_stop_path, model, in_dim, args)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_pr_auc={float(val_metrics['pr_auc']):.4f} "
            f"val_roc_auc={float(val_metrics['roc_auc']):.4f} "
            f"val_brier={float(val_metrics['brier']):.4f}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "early_stopping_triggered "
                f"monitor={args.early_stopping_monitor} "
                f"patience={args.early_stopping_patience} "
                f"best_monitor={best_monitor:.6f}"
            )
            break

    _save_checkpoint(last_path, model, in_dim, args)

    history_df = pd.DataFrame(history_rows)
    history_csv = output_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)
    _plot_losses(history_df, plot_dir / "loss_curves.png")

    candidates = {
        "early_stop_best": early_stop_path,
        "best_val_loss": best_loss_path,
        "best_val_pr_auc": best_pr_path,
        "last_epoch": last_path,
    }

    summary = {}
    for name, ckpt_path in candidates.items():
        if not ckpt_path.exists():
            print(f"skipping_missing_checkpoint name={name} path={ckpt_path}")
            continue
        model_eval = _load_model(ckpt_path, device)
        y_test, p_test = _collect_predictions(model_eval, test_loader, device)
        metrics = _compute_metrics(y_test, p_test, threshold=args.conf_threshold)
        summary[name] = metrics

        with open(metrics_dir / f"{name}_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        _plot_pr_curve(
            y_test,
            p_test,
            plot_dir / f"{name}_pr_curve.png",
            title=f"Precision-Recall ({name})",
        )
        _plot_log_roc(
            y_test,
            p_test,
            plot_dir / f"{name}_log_roc_curve.png",
            title=f"Log-ROC ({name})",
        )
        _plot_confusion_matrix(
            y_test,
            p_test,
            plot_dir / f"{name}_confusion_matrix.png",
            title=f"Confusion Matrix ({name})",
            threshold=args.conf_threshold,
        )

    with open(output_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved training history to: {history_csv}")
    print(f"Saved plots to: {plot_dir}")
    print(f"Saved test metrics to: {metrics_dir}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ProLinnator GNN with train/val/test evaluation")
    p.add_argument(
        "--task",
        choices=["experimental", "native_proline"],
        default="experimental",
        help="Task type: experimental mutability labels or native Proline residue pretraining labels.",
    )
    p.add_argument(
        "--mask-sequence-features",
        action="store_true",
        help="Mask sequence-derived features (AA one-hot and appended embeddings) while preserving structural features.",
    )
    p.add_argument("--csv", required=True, help="Path to mutation CSV")
    p.add_argument("--structures-dir", required=True, help="Directory containing PDB files")
    p.add_argument("--single-pdb", default=None, help="Optional single-protein mode PDB filename")
    p.add_argument("--single-chain", default="A", help="Chain ID for single-protein mode")
    p.add_argument("--ddg-threshold", type=float, default=1.0)
    p.add_argument("--distance-cutoff", type=float, default=8.0)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument(
        "--sample-mode",
        choices=["full_graph", "ego_subgraph"],
        default="full_graph",
        help="Training sample granularity: full protein graphs or residue-centered ego-subgraphs.",
    )
    p.add_argument(
        "--subgraph-hops",
        type=int,
        default=1,
        help="Number of hops for ego-subgraph sampling when --sample-mode ego_subgraph is used.",
    )

    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="Held-out test fraction. Set to 0.0 for train/val-only mode; final metrics are reported on val.")

    p.add_argument("--conf-threshold", type=float, default=0.5)

    p.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="focal",
        help="Training loss type.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for focal loss.",
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional positive-class alpha for focal loss (e.g., 0.25); unset disables alpha weighting.",
    )
    p.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Apply class-ratio positive weighting inside the selected loss.",
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

    p.add_argument(
        "--early-stopping-monitor",
        choices=["val_pr_auc", "val_loss"],
        default="val_pr_auc",
        help="Metric used for early stopping.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop if monitor does not improve for this many epochs (0 disables early stopping).",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-3,
        help="Minimum monitor improvement required to reset early stopping patience.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Optional checkpoint path for initialization before training/finetuning.",
    )
    p.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="If >0 and --pretrained-checkpoint is set, keep encoder frozen for the first N epochs.",
    )
    p.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for encoder params; head uses --lr.",
    )
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
