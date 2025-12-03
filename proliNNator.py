#!/usr/bin/env python3
from __future__ import annotations
print("""
                     _  _  _   _  _   _         _                
 _ __   _ __   ___  | |(_)| \ | || \ | |  __ _ | |_   ___   _ __ 
| '_ \ | '__| / _ \ | || ||  \| ||  \| | / _` || __| / _ \ | '__|
| |_) || |   | (_) || || || |\  || |\  || (_| || |_ | (_) || |   
| .__/ |_|    \___/ |_||_||_| \_||_| \_| \__,_| \__| \___/ |_|   
|_|                                                                               
""")

"""
Load a trained ResidueGAT model and annotate a PDB with proline probabilities.

The script parses a single PDB file, runs the classifier on each residue node,
and writes a new PDB where every atom in a residue receives the predicted
probability (0..1) in its B-factor column.
"""
# General imports
import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#Pytorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

#Biopython imports
from Bio.PDB import PDBIO, PDBParser, is_aa
from Bio.PDB.vectors import Vector, calc_dihedral


# ---------- Model definition (matches train_gcn.py) ----------

class ResidueGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0, heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv

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
        x = torch.relu(x)
        return self.mlp(x).squeeze(-1)


# ---------- Geometry utilities ----------

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def get_atom_vector(residue, atom_name: str) -> Optional[Vector]:
    if residue is None:
        return None
    if atom_name not in residue:
        return None
    return residue[atom_name].get_vector()


def dihedral_or_none(vectors: Sequence[Optional[Vector]]) -> Optional[float]:
    if any(vec is None for vec in vectors):
        return None
    return math.degrees(calc_dihedral(*vectors))


def torsion_to_cos_sin(angle_deg: Optional[float]) -> Tuple[float, float]:
    if angle_deg is None:
        return 0.0, 0.0
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def count_sidechain_heavy_atoms(residue) -> int:
    count = 0
    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        if name in BACKBONE_ATOMS:
            continue
        element = (atom.element or "").strip().upper()
        if not element:
            element = name[0].upper()
        if element == "H":
            continue
        count += 1
    return count


def chain_residues(chain) -> List:
    residues = []
    for residue in chain.get_residues():
        if not is_aa(residue, standard=True):
            continue
        if "CA" not in residue:
            continue
        residues.append(residue)
    return residues


def build_node_features(residue, prev_res, next_res) -> List[float]:
    phi = (
        dihedral_or_none(
            [
                get_atom_vector(prev_res, "C"),
                get_atom_vector(residue, "N"),
                get_atom_vector(residue, "CA"),
                get_atom_vector(residue, "C"),
            ]
        )
        if prev_res is not None
        else None
    )
    psi = (
        dihedral_or_none(
            [
                get_atom_vector(residue, "N"),
                get_atom_vector(residue, "CA"),
                get_atom_vector(residue, "C"),
                get_atom_vector(next_res, "N"),
            ]
        )
        if next_res is not None
        else None
    )
    omega = (
        dihedral_or_none(
            [
                get_atom_vector(residue, "CA"),
                get_atom_vector(residue, "C"),
                get_atom_vector(next_res, "N"),
                get_atom_vector(next_res, "CA"),
            ]
        )
        if next_res is not None
        else None
    )

    phi_c, phi_s = torsion_to_cos_sin(phi)
    psi_c, psi_s = torsion_to_cos_sin(psi)
    omg_c, omg_s = torsion_to_cos_sin(omega)
    sidechain = float(count_sidechain_heavy_atoms(residue))
    x, y, z = residue["CA"].coord
    return [
        phi_c,
        phi_s,
        psi_c,
        psi_s,
        omg_c,
        omg_s,
        sidechain,
        float(x),
        float(y),
        float(z),
    ]


def structure_to_graph(structure) -> Tuple[Data, List]:
    nodes = []
    edges = []
    residue_refs = []

    model = next(structure.get_models())
    for chain in model:
        residues = chain_residues(chain)
        prev_idx = None
        for idx, residue in enumerate(residues):
            prev_res = residues[idx - 1] if idx > 0 else None
            next_res = residues[idx + 1] if idx + 1 < len(residues) else None
            features = build_node_features(residue, prev_res, next_res)
            nodes.append(features)
            residue_refs.append(residue)
            node_idx = len(nodes) - 1
            if prev_idx is not None:
                edges.append((prev_idx, node_idx))
                edges.append((node_idx, prev_idx))
            prev_idx = node_idx

    x = torch.tensor(nodes, dtype=torch.float)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data, residue_refs


def annotate_structure(residue_refs: List, probabilities: torch.Tensor) -> None:
    for residue, prob in zip(residue_refs, probabilities.tolist()):
        for atom in residue.get_atoms():
            atom.set_bfactor(float(prob))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a PDB with proline probabilities from a trained GAT model."
    )
    parser.add_argument("--model-path", required=True, type=Path, help="Path to trained model (.pt).")
    parser.add_argument("--pdb-path", required=True, type=Path, help="Input PDB file.")
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Destination PDB with probabilities in B-factor column.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension used during training.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    pdb_path = args.pdb_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("inference", str(pdb_path))

    data, residue_refs = structure_to_graph(structure)
    if data.num_nodes == 0:
        raise RuntimeError("No residues with CA atoms found in the structure.")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    model = ResidueGAT(in_dim=data.num_features, hidden_dim=args.hidden_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu()

    annotate_structure(residue_refs, probs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    print(f"Wrote annotated PDB to {output_path}")


if __name__ == "__main__":
    main()

