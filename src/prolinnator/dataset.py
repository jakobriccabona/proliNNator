from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.vectors import Vector, calc_dihedral
from torch_geometric.data import Data

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def _to_canonical_columns(df: pd.DataFrame, require_supervision: bool = True) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        cu = c.strip().upper().replace(" ", "_")
        if cu in {"WT", "WILDTYPE", "WILD_TYPE"}:
            rename_map[c] = "WT"
        elif cu in {"POS", "POSITION", "RESID", "RESIDUE", "RESNUM", "RES_NUM"}:
            rename_map[c] = "POS"
        elif cu in {"MUT", "MUTANT", "MUT_TO_PRO", "MUTATION", "MUT_AA"}:
            rename_map[c] = "MUT"
        elif cu in {"DDG", "DELTADELTA_G", "DELTA_DELTA_G", "DDELG", "DDELTA_G"}:
            rename_map[c] = "DDG"
        elif cu in {"LABEL", "Y", "TARGET", "DMS_SCORE_BIN"}:
            rename_map[c] = "LABEL"
        elif cu in {"PROTEIN_ID", "PROTEIN", "TARGET_ID", "UNIPROT_ID", "DMS_ID"}:
            rename_map[c] = "protein_id"
        elif cu in {"PDB_FILE", "STRUCTURE", "STRUCTURE_FILE"}:
            rename_map[c] = "pdb_file"
        elif cu in {"CHAIN", "CHAIN_ID"}:
            rename_map[c] = "chain_id"

    out = df.rename(columns=rename_map).copy()

    required = {"WT", "POS", "MUT"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if require_supervision and "DDG" not in out.columns and "LABEL" not in out.columns:
        raise ValueError("Mutation table must contain either DDG or LABEL columns")

    out["WT"] = out["WT"].astype(str).str.upper().str.strip()
    out["MUT"] = out["MUT"].astype(str).str.upper().str.strip()
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")

    if "DDG" in out.columns:
        out["DDG"] = pd.to_numeric(out["DDG"], errors="coerce")
    if "LABEL" in out.columns:
        out["LABEL"] = pd.to_numeric(out["LABEL"], errors="coerce")

    keep_cols = ["POS"]
    if "DDG" in out.columns:
        keep_cols.append("DDG")
    if "LABEL" in out.columns:
        keep_cols.append("LABEL")

    out = out.dropna(subset=keep_cols)  # type: ignore[arg-type]
    out["POS"] = out["POS"].astype(int)

    if "chain_id" not in out.columns:
        out["chain_id"] = "A"
    out["chain_id"] = out["chain_id"].astype(str).str.strip().replace("", "A")

    return out


def _to_native_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        cu = c.strip().upper().replace(" ", "_")
        if cu in {"PROTEIN_ID", "PROTEIN", "TARGET_ID", "UNIPROT_ID", "DMS_ID"}:
            rename_map[c] = "protein_id"
        elif cu in {"PDB_FILE", "STRUCTURE", "STRUCTURE_FILE", "PDB"}:
            rename_map[c] = "pdb_file"
        elif cu in {"CHAIN", "CHAIN_ID"}:
            rename_map[c] = "chain_id"

    out = df.rename(columns=rename_map).copy()
    if "pdb_file" not in out.columns:
        raise ValueError("Native-proline pretraining CSV must include a PDB filename column (e.g. pdb_file)")

    if "protein_id" not in out.columns:
        out["protein_id"] = out["pdb_file"].astype(str).map(lambda x: Path(x).stem)

    if "chain_id" not in out.columns:
        out["chain_id"] = "A"
    out["chain_id"] = out["chain_id"].astype(str).str.strip().replace("", "A")
    out["pdb_file"] = out["pdb_file"].astype(str).str.strip()
    out["protein_id"] = out["protein_id"].astype(str).str.strip()
    out = out[(out["pdb_file"] != "") & (out["protein_id"] != "")]

    return out[["protein_id", "pdb_file", "chain_id"]].drop_duplicates().reset_index(drop=True)


def _aa_one_hot(aa: str) -> np.ndarray:
    vec = np.zeros(len(AA_ORDER), dtype=np.float32)
    idx = AA_TO_IDX.get(aa)
    if idx is not None:
        vec[idx] = 1.0
    return vec


@dataclass
class ResidueRecord:
    pdb_number: int
    aa: str
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    o: np.ndarray | None


def _extract_chain_residues(pdb_path: Path, chain_id: str) -> List[ResidueRecord]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    residues: List[ResidueRecord] = []
    for model in structure:
        if chain_id not in model:
            continue
        chain = model[chain_id]
        for residue in chain:
            hetflag, resseq, _icode = residue.id
            if hetflag.strip() != "":
                continue
            resname = residue.get_resname().upper()
            aa = three_to_one.get(resname)
            if aa is None or "N" not in residue or "CA" not in residue or "C" not in residue:
                continue
            n = residue["N"].get_coord().astype(np.float32)
            ca = residue["CA"].get_coord().astype(np.float32)
            c = residue["C"].get_coord().astype(np.float32)
            o = residue["O"].get_coord().astype(np.float32) if "O" in residue else None
            residues.append(ResidueRecord(pdb_number=int(resseq), aa=aa, n=n, ca=ca, c=c, o=o))
        break

    if not residues:
        raise ValueError(f"No residues extracted for chain '{chain_id}' in {pdb_path}")

    return residues


def _build_edges(coords: np.ndarray, distance_cutoff: float = 8.0) -> np.ndarray:
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=-1))

    edges = set()
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    idx_i, idx_j = np.where((dmat <= distance_cutoff) & (dmat > 0.0))
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        edges.add((i, j))

    return np.array(sorted(edges), dtype=np.int64).T


def _compute_backbone_torsion_features(residues: List[ResidueRecord]) -> np.ndarray:
    n = len(residues)
    # Features: [sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)]
    torsion = np.zeros((n, 6), dtype=np.float32)

    for i in range(n):
        phi = None
        psi = None
        omega = None

        if i > 0:
            try:
                phi = calc_dihedral(
                    Vector(residues[i - 1].c),
                    Vector(residues[i].n),
                    Vector(residues[i].ca),
                    Vector(residues[i].c),
                )
            except Exception:
                phi = None

        if i < n - 1:
            try:
                psi = calc_dihedral(
                    Vector(residues[i].n),
                    Vector(residues[i].ca),
                    Vector(residues[i].c),
                    Vector(residues[i + 1].n),
                )
            except Exception:
                psi = None

            try:
                omega = calc_dihedral(
                    Vector(residues[i].ca),
                    Vector(residues[i].c),
                    Vector(residues[i + 1].n),
                    Vector(residues[i + 1].ca),
                )
            except Exception:
                omega = None

        if phi is not None:
            torsion[i, 0] = np.sin(phi)
            torsion[i, 1] = np.cos(phi)

        if psi is not None:
            torsion[i, 2] = np.sin(psi)
            torsion[i, 3] = np.cos(psi)

        if omega is not None:
            torsion[i, 4] = np.sin(omega)
            torsion[i, 5] = np.cos(omega)

    return torsion


def _compute_backbone_n_hbond_feature(residues: List[ResidueRecord], distance_cutoff: float = 3.5) -> np.ndarray:
    n = len(residues)
    # Binary feature: 1 if backbone N is close to a non-neighbor backbone O, else 0.
    feat = np.zeros((n, 1), dtype=np.float32)

    oxygen_idx: List[int] = []
    oxygen_coords: List[np.ndarray] = []
    for idx, r in enumerate(residues):
        if r.o is not None:
            oxygen_idx.append(idx)
            oxygen_coords.append(r.o)

    if not oxygen_coords:
        return feat

    o_arr = np.stack(oxygen_coords, axis=0)
    o_idx_arr = np.array(oxygen_idx, dtype=np.int64)

    for i, r in enumerate(residues):
        d = np.linalg.norm(o_arr - r.n[None, :], axis=1)
        non_neighbor = np.abs(o_idx_arr - i) > 1
        if np.any((d <= distance_cutoff) & non_neighbor):
            feat[i, 0] = 1.0

    return feat


def _assign_labels(
    residues: List[ResidueRecord],
    mutation_df: pd.DataFrame,
    ddg_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_to_idx = {r.pdb_number: idx for idx, r in enumerate(residues)}

    labels = np.zeros(len(residues), dtype=np.float32)
    mask = np.zeros(len(residues), dtype=bool)
    use_label = "LABEL" in mutation_df.columns

    for _, row in mutation_df.iterrows():
        pos = int(row["POS"])
        if pos not in pos_to_idx:
            continue
        idx = pos_to_idx[pos]

        wt = str(row["WT"]).upper().strip()
        if wt and wt in AA_TO_IDX and wt != residues[idx].aa:
            continue

        y = float(row["LABEL"]) if use_label else (1.0 if float(row["DDG"]) <= ddg_threshold else 0.0)
        if y < 0.0 or y > 1.0:
            continue

        labels[idx] = y
        mask[idx] = True

    return labels, mask


def _resolve_embedding_file(embedding_dir: Path, protein_id: str, pdb_file: str) -> Path | None:
    pdb_stem = Path(pdb_file).stem
    candidates = [
        embedding_dir / f"{protein_id}.npy",
        embedding_dir / f"{protein_id}.npz",
        embedding_dir / f"{pdb_stem}.npy",
        embedding_dir / f"{pdb_stem}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_embeddings_for_residues(
    embedding_path: Path,
    residue_numbers: np.ndarray,
    expected_len: int,
) -> np.ndarray:
    if embedding_path.suffix.lower() == ".npy":
        emb = np.load(embedding_path)
        if emb.ndim != 2:
            raise ValueError(f"Embedding file must be 2D: {embedding_path}")
        if emb.shape[0] != expected_len:
            raise ValueError(
                f"Embedding length mismatch for {embedding_path}: got {emb.shape[0]}, expected {expected_len}. "
                "Use NPZ with residue_numbers for explicit mapping if sequence order differs."
            )
        return emb.astype(np.float32)

    if embedding_path.suffix.lower() == ".npz":
        blob = np.load(embedding_path)
        if "embeddings" not in blob:
            raise ValueError(f"NPZ must contain key 'embeddings': {embedding_path}")
        emb = blob["embeddings"]
        if emb.ndim != 2:
            raise ValueError(f"NPZ embeddings must be 2D: {embedding_path}")

        if emb.shape[0] == expected_len:
            return emb.astype(np.float32)

        if "residue_numbers" not in blob:
            raise ValueError(
                f"Cannot map embeddings in {embedding_path}: embeddings length {emb.shape[0]} != expected {expected_len} "
                "and 'residue_numbers' key is missing."
            )

        emb_resnums = blob["residue_numbers"].astype(int)
        if emb_resnums.shape[0] != emb.shape[0]:
            raise ValueError(f"residue_numbers and embeddings length mismatch in {embedding_path}")

        pos_to_vec = {int(r): emb[i] for i, r in enumerate(emb_resnums.tolist())}
        rows = []
        for r in residue_numbers.tolist():
            if int(r) not in pos_to_vec:
                raise ValueError(f"Missing embedding for residue number {int(r)} in {embedding_path}")
            rows.append(pos_to_vec[int(r)])
        return np.stack(rows, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported embedding file type: {embedding_path}")


def build_graph_data_for_protein(
    mutation_df: pd.DataFrame,
    pdb_path: Path,
    pdb_file: str,
    chain_id: str,
    protein_id: str,
    ddg_threshold: float,
    distance_cutoff: float,
    embedding_dir: Path | None,
    embedding_strict: bool,
    task: str = "experimental",
    mask_sequence_features: bool = False,
) -> Data:
    residues = _extract_chain_residues(pdb_path, chain_id)
    coords = np.stack([r.ca for r in residues], axis=0)
    residue_numbers = np.array([r.pdb_number for r in residues], dtype=int)
    aa_feats = np.stack([_aa_one_hot(r.aa) for r in residues], axis=0)
    torsion_feats = _compute_backbone_torsion_features(residues)
    hbond_n_feat = _compute_backbone_n_hbond_feature(residues)
    aa_feats = np.concatenate([aa_feats, torsion_feats, hbond_n_feat], axis=1)

    if embedding_dir is not None:
        emb_path = _resolve_embedding_file(embedding_dir, protein_id=protein_id, pdb_file=pdb_file)
        if emb_path is None:
            if embedding_strict:
                raise FileNotFoundError(
                    f"No embedding file found for protein_id={protein_id} / pdb_file={pdb_file} in {embedding_dir}"
                )
        else:
            emb = _load_embeddings_for_residues(
                embedding_path=emb_path,
                residue_numbers=residue_numbers,
                expected_len=aa_feats.shape[0],
            )
            aa_feats = np.concatenate([aa_feats, emb], axis=1)

    if mask_sequence_features:
        # Keep dimensionality intact while removing sequence-derived content.
        aa_feats[:, :20] = 0.0
        if aa_feats.shape[1] > 27:
            aa_feats[:, 27:] = 0.0

    edge_index = _build_edges(coords, distance_cutoff=distance_cutoff)
    if task == "native_proline":
        labels = np.array([1.0 if r.aa == "P" else 0.0 for r in residues], dtype=np.float32)
        mask = np.ones(len(residues), dtype=bool)
    else:
        labels, mask = _assign_labels(residues, mutation_df, ddg_threshold=ddg_threshold)

    data = Data(
        x=torch.tensor(aa_feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.float32),
        label_mask=torch.tensor(mask, dtype=torch.bool),
    )
    data.protein_id = protein_id
    data.chain_id = chain_id
    data.residue_numbers = torch.tensor(residue_numbers, dtype=torch.long)
    data.ca_coords = torch.tensor(coords, dtype=torch.float32)
    return data


def load_dataset(
    csv_path: Path,
    structures_dir: Path,
    ddg_threshold: float = 1.0,
    distance_cutoff: float = 8.0,
    single_pdb: str | None = None,
    single_chain: str = "A",
    embeddings_dir: Path | None = None,
    embedding_strict: bool = False,
    task: str = "experimental",
    mask_sequence_features: bool = False,
) -> List[Data]:
    if task not in {"experimental", "native_proline"}:
        raise ValueError(f"Unsupported task: {task}")

    df = pd.read_csv(csv_path)
    if task == "experimental":
        df = _to_canonical_columns(df, require_supervision=True)
        df = df[df["MUT"].str.startswith("P")].copy()
        if df.empty:
            raise ValueError("No Proline mutations found after filtering on MUT == P")
    else:
        df = _to_native_index_columns(df)

    dataset: List[Data] = []

    if single_pdb is not None:
        protein_id = "single_protein"
        pdb_path = structures_dir / single_pdb
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        data = build_graph_data_for_protein(
            mutation_df=df,
            pdb_path=pdb_path,
            pdb_file=single_pdb,
            chain_id=single_chain,
            protein_id=protein_id,
            ddg_threshold=ddg_threshold,
            distance_cutoff=distance_cutoff,
            embedding_dir=embeddings_dir,
            embedding_strict=embedding_strict,
            task=task,
            mask_sequence_features=mask_sequence_features,
        )
        dataset.append(data)
        return dataset

    if "protein_id" not in df.columns or "pdb_file" not in df.columns:
        raise ValueError(
            "For multi-protein mode, CSV must include 'protein_id' and 'pdb_file' columns, "
            "or use --single-pdb for single-protein mode."
        )

    grouped = df.groupby(["protein_id", "pdb_file", "chain_id"], dropna=False)
    for (protein_id, pdb_file, chain_id), gdf in grouped:
        pdb_path = structures_dir / str(pdb_file)
        if not pdb_path.exists():
            continue
        try:
            data = build_graph_data_for_protein(
                mutation_df=gdf,
                pdb_path=pdb_path,
                pdb_file=str(pdb_file),
                chain_id=str(chain_id),
                protein_id=str(protein_id),
                ddg_threshold=ddg_threshold,
                distance_cutoff=distance_cutoff,
                embedding_dir=embeddings_dir,
                embedding_strict=embedding_strict,
                task=task,
                mask_sequence_features=mask_sequence_features,
            )
        except Exception:
            continue
        if int(data.label_mask.sum().item()) == 0:
            continue
        dataset.append(data)

    if not dataset:
        raise ValueError("No valid protein graphs were created. Check CSV mapping and PDB files.")

    return dataset
