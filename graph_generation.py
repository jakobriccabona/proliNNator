#!/usr/bin/env python3
"""
Graph generation utilities for protein structures.

Each amino acid residue becomes a node positioned at the C-alpha atom with
backbone torsions (phi, psi, omega) and sidechain heavy-atom counts as features.
Edges connect residues that share a peptide bond (sequence-adjacent residues).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.vectors import Vector, calc_dihedral


@dataclass
class ResidueNode:
    """Node representation for a residue."""

    chain_id: str
    residue_id: str
    residue_name: str
    ca_coord: Tuple[float, float, float]
    phi: Optional[float]
    psi: Optional[float]
    omega: Optional[float]
    sidechain_heavy_atoms: int
    label: int


@dataclass
class ResidueGraph:
    """Graph containing nodes and peptide-bond edges."""

    structure_id: str
    nodes: List[ResidueNode]
    edges: List[Tuple[int, int]]


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate residue graphs with torsion features."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory with protein structure files (PDB/mmCIF).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Destination JSONL file for serialized graphs.",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional upper bound on processed structures (for debugging).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdb", ".ent", ".cif", ".mmcif"],
        help="File extensions to consider when scanning dataset_dir.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_structure_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield structure files under root matching given extensions."""
    normalized = tuple(ext.lower() for ext in extensions)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in normalized:
            yield path


def parse_structure(path: Path, pdb_parser: PDBParser, cif_parser: MMCIFParser):
    """Parse a structure file into a Bio.PDB Structure object."""
    suffix = path.suffix.lower()
    structure_id = path.stem
    if suffix in {".cif", ".mmcif"}:
        return cif_parser.get_structure(structure_id, str(path))
    return pdb_parser.get_structure(structure_id, str(path))


def residue_id(residue) -> str:
    """Create a stable identifier for a residue."""
    hetflag, seq_id, insertion = residue.id
    insertion = insertion.strip() or "-"
    return f"{hetflag}{seq_id}{insertion}"


def get_atom_vector(residue, atom_name: str) -> Optional[Vector]:
    """Return the atom vector if present, else None."""
    if residue is None:
        return None
    if atom_name not in residue:
        return None
    return residue[atom_name].get_vector()


def dihedral_or_none(vectors: Sequence[Optional[Vector]]) -> Optional[float]:
    """Compute dihedral angle in degrees when all vectors are present."""
    if any(vec is None for vec in vectors):
        return None
    angle = calc_dihedral(*vectors)
    return math.degrees(angle)


def count_sidechain_heavy_atoms(residue) -> int:
    """Count non-hydrogen atoms that are not part of the backbone."""
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
    """Filter residues in a chain to standard amino acids with CA atoms."""
    residues = []
    for residue in chain.get_residues():
        if not is_aa(residue, standard=True):
            continue
        if "CA" not in residue:
            continue
        residues.append(residue)
    return residues


def residues_to_graph(structure, structure_id: str) -> ResidueGraph:
    """Convert a parsed structure into a residue graph."""
    nodes: List[ResidueNode] = []
    edges: List[Tuple[int, int]] = []

    # Use the first model; many structures only contain one
    model = next(structure.get_models())
    for chain in model:
        residues = chain_residues(chain)
        prev_node_index: Optional[int] = None
        for idx, residue in enumerate(residues):
            prev_res = residues[idx - 1] if idx > 0 else None
            next_res = residues[idx + 1] if idx + 1 < len(residues) else None

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

            ca_atom = residue["CA"]
            node = ResidueNode(
                chain_id=chain.id,
                residue_id=residue_id(residue),
                residue_name=residue.resname,
                ca_coord=tuple(float(coord) for coord in ca_atom.coord),
                phi=phi,
                psi=psi,
                omega=omega,
                sidechain_heavy_atoms=count_sidechain_heavy_atoms(residue),
                label=int(residue.resname.strip().upper() == "PRO"),
            )
            node_index = len(nodes)
            nodes.append(node)

            if prev_node_index is not None:
                edges.append((prev_node_index, node_index))

            prev_node_index = node_index

    return ResidueGraph(structure_id=structure_id, nodes=nodes, edges=edges)


def serialize_graph(graph: ResidueGraph) -> str:
    """Serialize a graph dataclass as JSON."""
    return json.dumps(
        {
            "structure_id": graph.structure_id,
            "nodes": [asdict(node) for node in graph.nodes],
            "edges": graph.edges,
        }
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    files = iter_structure_files(dataset_dir, args.extensions)

    processed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for structure_path in files:
            if args.max_structures is not None and processed >= args.max_structures:
                break
            try:
                structure = parse_structure(structure_path, pdb_parser, cif_parser)
                graph = residues_to_graph(structure, structure_path.stem)
            except Exception as exc:  # pragma: no cover - logging path
                logging.warning("Failed to process %s (%s)", structure_path, exc)
                continue

            if not graph.nodes:
                logging.debug("Skipping %s - no residues with CA atoms", structure_path)
                continue

            handle.write(f"{serialize_graph(graph)}\n")
            processed += 1
            if processed % 100 == 0:
                logging.info("Processed %d structures", processed)

    logging.info("Finished. Structures processed: %d", processed)


if __name__ == "__main__":
    main()

