from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from prolinnator.dataset import load_dataset
from prolinnator.model import ProlineSiteGNN


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = load_dataset(
        csv_path=Path(args.csv),
        structures_dir=Path(args.structures_dir),
        ddg_threshold=args.ddg_threshold,
        distance_cutoff=args.distance_cutoff,
        single_pdb=args.single_pdb,
        single_chain=args.single_chain,
    )
    if len(dataset) != 1:
        raise ValueError("Prediction script expects single graph. Use single-protein inputs.")

    graph = dataset[0].to(device)

    ckpt = torch.load(args.model, map_location=device)
    model = ProlineSiteGNN(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()

    out = pd.DataFrame(
        {
            "protein_id": [graph.protein_id] * len(probs),
            "chain_id": [graph.chain_id] * len(probs),
            "residue_number": graph.residue_numbers.cpu().numpy(),
            "proline_mutability_probability": probs,
            "has_experimental_label": graph.label_mask.cpu().numpy().astype(int),
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict per-residue Proline mutability probabilities")
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--single-pdb", required=True)
    p.add_argument("--single-chain", default="A")
    p.add_argument("--ddg-threshold", type=float, default=1.0)
    p.add_argument("--distance-cutoff", type=float, default=8.0)
    p.add_argument("--out", default="data/processed/probabilities.csv")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    parser = make_parser()
    run(parser.parse_args())
