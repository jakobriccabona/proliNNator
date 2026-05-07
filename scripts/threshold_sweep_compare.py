from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from prolinnator.dataset import load_dataset
from prolinnator.train import _collect_predictions, _load_model, _split_dataset
from prolinnator.train_tabular import _extract_labeled_arrays


def eval_at(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), pred, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": float(thr),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, name: str) -> dict:
    thresholds = np.linspace(0.05, 0.95, 181)
    val_rows = [eval_at(y_val, p_val, t) for t in thresholds]
    best = max(val_rows, key=lambda d: (d["f1"], d["recall"]))
    return {
        "model": name,
        "best_val_threshold_by_f1": best,
        "test_at_0.5": eval_at(y_test, p_test, 0.5),
        "test_at_tuned_threshold": eval_at(y_test, p_test, best["threshold"]),
    }


def main() -> None:
    dataset = load_dataset(
        csv_path=Path("data/processed/proline_training_from_proteingym_aligned.csv"),
        structures_dir=Path("data/raw/structures/ProteinGym_AF2_structures"),
        embeddings_dir=Path("data/processed/embeddings_esm2_t33"),
        embedding_strict=True,
    )
    train_set, val_set, test_set = _split_dataset(dataset, 0.7, 0.15, 0.15, seed=42)

    # GNN from saved checkpoint
    ckpt = Path("outputs/run_2026-05-05_500ep_aligned_esm2_bce_omega_hbond/checkpoints/early_stop_best.pt")
    model = _load_model(ckpt, torch.device("cpu"))
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    y_val_gnn, p_val_gnn = _collect_predictions(model, val_loader, torch.device("cpu"))
    y_test_gnn, p_test_gnn = _collect_predictions(model, test_loader, torch.device("cpu"))

    # CatBoost retrain on same split
    x_train, y_train, _ = _extract_labeled_arrays(train_set)
    x_val, y_val_tab, _ = _extract_labeled_arrays(val_set)
    x_test, y_test_tab, _ = _extract_labeled_arrays(test_set)

    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=1.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
    cat.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val_tab)])
    p_val_cat = cat.predict_proba(x_val)[:, 1]
    p_test_cat = cat.predict_proba(x_test)[:, 1]

    assert np.array_equal(y_val_tab, y_val_gnn.astype(int))
    assert np.array_equal(y_test_tab, y_test_gnn.astype(int))

    out = {
        "split": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "objective": "maximize validation F1 over thresholds 0.05..0.95",
        "gnn": sweep(y_val_gnn, p_val_gnn, y_test_gnn, p_test_gnn, "gnn_bce_omega_hbond"),
        "catboost": sweep(y_val_tab, p_val_cat, y_test_tab, p_test_cat, "catboost_omega_hbond"),
    }

    out_path = Path("outputs/threshold_sweep_omega_hbond_f1.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"written={out_path}")


if __name__ == "__main__":
    main()
