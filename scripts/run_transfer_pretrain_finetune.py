from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from prolinnator.train import make_parser, train


def _ns_from_cli(args_list: list[str]) -> argparse.Namespace:
    parser = make_parser()
    args = parser.parse_args(args_list)
    return args


def _pick_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    candidates = [
        ckpt_dir / "early_stop_best.pt",
        ckpt_dir / "best_val_pr_auc.pt",
        ckpt_dir / "best_val_loss.pt",
        ckpt_dir / "last_epoch.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "test_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_training(args_list: list[str], run_dir: Path, reuse_existing: bool) -> None:
    summary_path = run_dir / "test_summary.json"
    if reuse_existing and summary_path.exists():
        print(f"reuse_existing run_dir={run_dir}")
        return

    print("running", " ".join(args_list))
    args = _ns_from_cli(args_list)
    train(args)


def _extract_primary(metrics_blob: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["early_stop_best", "best_val_pr_auc", "best_val_loss", "last_epoch"]:
        if key in metrics_blob:
            out = dict(metrics_blob[key])
            out["checkpoint_name"] = key
            return out
    return {}


def build_orchestrator_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run native-Proline pretraining, experimental finetuning from pretrained checkpoint, "
            "and a matched from-scratch baseline; write a consolidated comparison report."
        )
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--embeddings-dir", default=None)
    p.add_argument("--embedding-strict", action="store_true")

    p.add_argument("--output-root", default="outputs/transfer_runs")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--pretrain-epochs", type=int, default=80)
    p.add_argument("--finetune-epochs", type=int, default=200)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--use-pos-weight", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None)

    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)

    p.add_argument("--sample-mode", choices=["full_graph", "ego_subgraph"], default="full_graph")
    p.add_argument("--subgraph-hops", type=int, default=1)

    p.add_argument("--freeze-encoder-epochs", type=int, default=5)
    p.add_argument("--encoder-lr-scale", type=float, default=0.3)
    p.add_argument("--mask-sequence-features", action="store_true", default=True)
    p.add_argument("--no-mask-sequence-features", dest="mask_sequence_features", action="store_false")

    p.add_argument("--early-stopping-monitor", choices=["val_pr_auc", "val_loss"], default="val_pr_auc")
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-3)

    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--reuse-existing", action="store_true")
    return p


def _common_train_flags(args: argparse.Namespace, epochs: int, output_dir: Path) -> list[str]:
    flags = [
        "--csv", args.csv,
        "--structures-dir", args.structures_dir,
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--loss", args.loss,
        "--focal-gamma", str(args.focal_gamma),
        "--train-fraction", str(args.train_fraction),
        "--val-fraction", str(args.val_fraction),
        "--test-fraction", str(args.test_fraction),
        "--sample-mode", args.sample_mode,
        "--subgraph-hops", str(args.subgraph_hops),
        "--early-stopping-monitor", args.early_stopping_monitor,
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--conf-threshold", str(args.conf_threshold),
        "--seed", str(args.seed),
        "--output-dir", str(output_dir),
    ]

    if args.embeddings_dir:
        flags.extend(["--embeddings-dir", args.embeddings_dir])
    if args.embedding_strict:
        flags.append("--embedding-strict")
    if args.use_pos_weight:
        flags.append("--use-pos-weight")
    if args.focal_alpha is not None:
        flags.extend(["--focal-alpha", str(args.focal_alpha)])

    return flags


def main() -> None:
    parser = build_orchestrator_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_pretrain = output_root / "stage1_native_pretrain"
    run_finetune = output_root / "stage2_finetune_from_pretrain"
    run_scratch = output_root / "stage2_scratch_baseline"

    pretrain_flags = _common_train_flags(args, epochs=args.pretrain_epochs, output_dir=run_pretrain)
    pretrain_flags.extend(["--task", "native_proline"])
    if args.mask_sequence_features:
        pretrain_flags.append("--mask-sequence-features")

    _run_training(pretrain_flags, run_pretrain, reuse_existing=args.reuse_existing)
    pretrained_ckpt = _pick_checkpoint(run_pretrain)

    finetune_flags = _common_train_flags(args, epochs=args.finetune_epochs, output_dir=run_finetune)
    finetune_flags.extend(
        [
            "--task", "experimental",
            "--pretrained-checkpoint", str(pretrained_ckpt),
            "--freeze-encoder-epochs", str(args.freeze_encoder_epochs),
            "--encoder-lr-scale", str(args.encoder_lr_scale),
        ]
    )

    _run_training(finetune_flags, run_finetune, reuse_existing=args.reuse_existing)

    scratch_flags = _common_train_flags(args, epochs=args.finetune_epochs, output_dir=run_scratch)
    scratch_flags.extend(["--task", "experimental"])
    _run_training(scratch_flags, run_scratch, reuse_existing=args.reuse_existing)

    pre_summary = _load_summary(run_pretrain)
    ft_summary = _load_summary(run_finetune)
    sc_summary = _load_summary(run_scratch)

    report = {
        "runs": {
            "stage1_native_pretrain": str(run_pretrain),
            "stage2_finetune_from_pretrain": str(run_finetune),
            "stage2_scratch_baseline": str(run_scratch),
        },
        "config": {
            "seed": args.seed,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "use_pos_weight": args.use_pos_weight,
            "sample_mode": args.sample_mode,
            "subgraph_hops": args.subgraph_hops,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
            "encoder_lr_scale": args.encoder_lr_scale,
            "mask_sequence_features": args.mask_sequence_features,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        },
        "stage1_primary_metrics": _extract_primary(pre_summary),
        "stage2_finetune_primary_metrics": _extract_primary(ft_summary),
        "stage2_scratch_primary_metrics": _extract_primary(sc_summary),
        "raw_test_summaries": {
            "stage1_native_pretrain": pre_summary,
            "stage2_finetune_from_pretrain": ft_summary,
            "stage2_scratch_baseline": sc_summary,
        },
    }

    ft = report["stage2_finetune_primary_metrics"]
    sc = report["stage2_scratch_primary_metrics"]
    if ft and sc:
        report["delta_finetune_minus_scratch"] = {
            "pr_auc": float(ft.get("pr_auc", 0.0) - sc.get("pr_auc", 0.0)),
            "roc_auc": float(ft.get("roc_auc", 0.0) - sc.get("roc_auc", 0.0)),
            "brier": float(ft.get("brier", 0.0) - sc.get("brier", 0.0)),
        }

    report_path = output_root / "transfer_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"written={report_path}")


if __name__ == "__main__":
    main()
