#!/usr/bin/env python3
"""
Train a graph attention network with an MLP head to classify proline residues.

Input graphs must come from graph_generation.py (JSONL). The script splits the
structures into train/validation/test sets, builds PyTorch Geometric batches,
and trains a GCN with three convolutional layers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GAT+MLP node classifier on residue graphs.")
    parser.add_argument(
        "--graph-file",
        required=True,
        type=Path,
        help="Path to graphs.jsonl produced by graph_generation.py",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Graphs per batch.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden feature size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--kernel-l2",
        type=float,
        default=0.0,
        help="Additional L2 penalty applied to GAT/MLP weights (set >0 to enable).",
    )
    parser.add_argument(
        "--train-val-test",
        nargs=3,
        type=float,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Fractions for dataset split (must sum to 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda|cpu).")
    parser.add_argument("--model-out", type=Path, default=None, help="Optional path to save model.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop training if val loss does not improve for these epochs (0 disables).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional PNG path to store training curves, confusion matrix, and PR curve.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torsion_to_cos_sin(angle_deg: float | None) -> Tuple[float, float]:
    if angle_deg is None:
        return 0.0, 0.0
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def build_node_features(node: dict) -> List[float]:
    phi_c, phi_s = torsion_to_cos_sin(node.get("phi"))
    psi_c, psi_s = torsion_to_cos_sin(node.get("psi"))
    omg_c, omg_s = torsion_to_cos_sin(node.get("omega"))
    sidechain = float(node.get("sidechain_heavy_atoms", 0))
    x, y, z = node.get("ca_coord", (0.0, 0.0, 0.0))
    return [phi_c, phi_s, psi_c, psi_s, omg_c, omg_s, sidechain, float(x), float(y), float(z)]


def graph_entry_to_data(entry: dict) -> Data:
    nodes = entry["nodes"]
    edges = entry["edges"]

    x = torch.tensor([build_node_features(node) for node in nodes], dtype=torch.float)
    y = torch.tensor([float(node.get("label", 0)) for node in nodes], dtype=torch.float)

    if edges:
        edge_pairs = []
        for src, dst in edges:
            edge_pairs.append((src, dst))
            edge_pairs.append((dst, src))
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)
    data.structure_id = entry.get("structure_id", "")
    return data


class ResidueGraphDataset(Dataset):
    def __init__(self, graph_file: Path):
        super().__init__()
        self.graphs: List[Data] = []
        with graph_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                data = graph_entry_to_data(entry)
                if data.x.size(0) == 0:
                    continue
                self.graphs.append(data)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]


def compute_class_counts(dataset: ResidueGraphDataset) -> Tuple[int, int]:
    pos = 0
    total = 0
    for data in dataset.graphs:
        pos += int(data.y.sum().item())
        total += data.num_nodes
    neg = total - pos
    return pos, neg


class ResidueGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.gat(x, edge_index)
        x = F.relu(x)
        return self.mlp(x).squeeze(-1)


def split_dataset(dataset: Dataset, splits: Sequence[float], seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-4):
        raise ValueError("Split fractions must sum to 1.0")
    lengths = [int(len(dataset) * frac) for frac in splits]
    remainder = len(dataset) - sum(lengths)
    lengths[0] += remainder  # assign leftover to train split
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, lengths, generator=generator)


def train_epoch(model, loader, criterion, optimizer, device, kernel_l2: float = 0.0):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        if kernel_l2 > 0:
            l2_reg = 0.0
            for param in model.parameters():
                if param.dim() > 1:
                    l2_reg = l2_reg + param.pow(2).sum()
            loss = loss + kernel_l2 * l2_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = criterion(logits, data.y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_loss += loss.item() * data.num_nodes
        total_correct += (preds == data.y).sum().item()
        total_nodes += data.num_nodes
    avg_loss = total_loss / max(total_nodes, 1)
    accuracy = total_correct / max(total_nodes, 1)
    return avg_loss, accuracy


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        prob = torch.sigmoid(logits).cpu()
        probs.append(prob)
        labels.append(data.y.cpu())
    if not probs:
        return np.array([]), np.array([])
    probs_tensor = torch.cat(probs).numpy()
    labels_tensor = torch.cat(labels).numpy()
    return probs_tensor, labels_tensor


def plot_diagnostics(history, confusion_norm, precision, recall, report_path: Path | None):
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax_loss = axes[0]
    ax_loss.plot(history["train_loss"], label="Train Loss")
    ax_loss.plot(history["val_loss"], label="Val Loss")
    ax_loss.set_xlabel("Epoch", fontsize=15)
    ax_loss.set_ylabel("Loss", fontsize=15)
    ax_loss.legend()
    ax_loss.set_title("Training vs Validation Loss")

    ax_pr = axes[1]
    ax_pr.plot(recall, precision, color="purple")
    ax_pr.set_xlabel("Recall", fontsize=15)
    ax_pr.set_ylabel("Precision", fontsize=15)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_title("Precision-Recall Curve (Test)")

    ax_cm_norm = axes[2]
    im_norm = ax_cm_norm.imshow(confusion_norm, cmap="Greens", vmin=0.0, vmax=1.0)
    ax_cm_norm.set_xticks([0, 1])
    ax_cm_norm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm_norm.set_yticks([0, 1])
    ax_cm_norm.set_yticklabels(["True 0", "True 1"])
    for i in range(confusion_norm.shape[0]):
        for j in range(confusion_norm.shape[1]):
            ax_cm_norm.text(
                j,
                i,
                f"{confusion_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    ax_cm_norm.set_title("Normalized Confusion Matrix (Test)")
    fig.colorbar(im_norm, ax=ax_cm_norm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_path, dpi=300)
        print(f"Saved diagnostics plot to {report_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    graph_file = args.graph_file.expanduser().resolve()
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    dataset = ResidueGraphDataset(graph_file)
    if dataset.len() == 0:
        raise RuntimeError("Dataset is empty. Check graph_generation.py output.")
    pos_count, neg_count = compute_class_counts(dataset)

    train_split, val_split, test_split = split_dataset(dataset, args.train_val_test, args.seed)

    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=args.batch_size)
    test_loader = DataLoader(test_split, batch_size=args.batch_size)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float, device=device)
    model = ResidueGAT(in_dim=dataset[0].num_features, hidden_dim=args.hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_since_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, kernel_l2=args.kernel_l2)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {args.early_stop_patience} epochs).")
                break

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss={test_loss:.4f} | acc={test_acc:.4f}")

    test_probs, test_labels = collect_predictions(model, test_loader, device)
    if test_probs.size and test_labels.size:
        test_preds = (test_probs >= 0.5).astype(int)
        cm_norm = confusion_matrix(test_labels, test_preds, normalize="true")
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        plot_diagnostics(history, cm_norm, precision, recall, args.report_path)

    if args.model_out is not None:
        model_path = args.model_out.expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
