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
from train_gcn import ResidueGAT

# ---------- Geometry utilities ----------

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# Hydrogen-bond detection threshold (N...O distance in Angstroms)
HBOND_DISTANCE = 3.5


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


def build_node_features(residue, prev_res, next_res, hbond_donors: int, hbond_acceptors: int) -> List[float]:
    """Build node features matching the training pipeline.

    Returns a 9-element feature vector:
    [phi_cos, phi_sin, psi_cos, psi_sin, omega_cos, omega_sin,
     sidechain_heavy_atoms, backbone_hbond_donors, backbone_hbond_acceptors]
    """
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
    return [
        phi_c,
        phi_s,
        psi_c,
        psi_s,
        omg_c,
        omg_s,
        sidechain,
        float(hbond_donors),
        float(hbond_acceptors),
    ]


def structure_to_graph(structure) -> Tuple[Data, List]:
    nodes = []
    edges = []
    residue_refs = []

    model = next(structure.get_models())
    # First pass: collect filtered residues per chain and add sequence adjacency
    for chain in model:
        residues = chain_residues(chain)
        prev_idx = None
        for residue in residues:
            residue_refs.append(residue)
            node_idx = len(residue_refs) - 1
            if prev_idx is not None:
                edges.append((prev_idx, node_idx))
                edges.append((node_idx, prev_idx))
            prev_idx = node_idx

    # Compute N and O vectors for all filtered residues
    n_vecs = [get_atom_vector(r, "N") for r in residue_refs]
    o_vecs = [get_atom_vector(r, "O") or get_atom_vector(r, "OXT") for r in residue_refs]

    # Initialize donor/acceptor counts
    donors = [0 for _ in residue_refs]
    acceptors = [0 for _ in residue_refs]

    # Iterate pairs to detect simple backbone H-bond like interactions
    num_nodes = len(residue_refs)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ni = n_vecs[i]
            oj = o_vecs[j]
            if ni is not None and oj is not None:
                if (ni - oj).norm() <= HBOND_DISTANCE:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
                    donors[i] += 1
                    acceptors[j] += 1

            nj = n_vecs[j]
            oi = o_vecs[i]
            if nj is not None and oi is not None:
                if (nj - oi).norm() <= HBOND_DISTANCE:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
                    donors[j] += 1
                    acceptors[i] += 1

    # Build node features now that donor/acceptor counts are known
    for idx, residue in enumerate(residue_refs):
        prev_res = residue_refs[idx - 1] if idx > 0 else None
        next_res = residue_refs[idx + 1] if idx + 1 < len(residue_refs) else None
        features = build_node_features(residue, prev_res, next_res, donors[idx], acceptors[idx])
        nodes.append(features)

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

