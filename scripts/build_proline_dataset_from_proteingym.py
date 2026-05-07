from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser

SINGLE_SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def parse_single_mutant(mut: str):
    m = SINGLE_SUB_RE.match(mut.strip().upper())
    if not m:
        return None
    wt, pos, aa = m.group(1), int(m.group(2)), m.group(3)
    return wt, pos, aa


def extract_pdb_sequence_and_resnums(pdb_path: Path, chain_id: str) -> tuple[str, list[int]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    seq_chars: list[str] = []
    resnums: list[int] = []
    for model in structure:
        if chain_id not in model:
            continue
        chain = model[chain_id]
        for residue in chain:
            hetflag, resseq, _icode = residue.id
            if hetflag.strip() != "":
                continue
            aa = THREE_TO_ONE.get(residue.get_resname().upper())
            if aa is None:
                continue
            seq_chars.append(aa)
            resnums.append(int(resseq))
        break

    if not seq_chars:
        raise ValueError(f"No sequence extracted from {pdb_path} chain {chain_id}")

    return "".join(seq_chars), resnums


def build_dms_to_pdb_pos_map(dms_seq: str, pdb_seq: str, pdb_resnums: list[int]) -> dict[int, int]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(dms_seq, pdb_seq)
    if len(alignments) == 0:
        return {}

    aln = alignments[0]
    dms_blocks = aln.aligned[0]
    pdb_blocks = aln.aligned[1]

    pos_map: dict[int, int] = {}
    for (d_start, d_end), (p_start, p_end) in zip(dms_blocks.tolist(), pdb_blocks.tolist()):
        block_len = min(d_end - d_start, p_end - p_start)
        for k in range(block_len):
            d_idx = d_start + k
            p_idx = p_start + k
            # Keep only identity matches for safe coordinate transfer.
            if dms_seq[d_idx] != pdb_seq[p_idx]:
                continue
            dms_pos_1based = d_idx + 1
            pos_map[dms_pos_1based] = int(pdb_resnums[p_idx])

    return pos_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Proline-only training CSV from ProteinGym substitution files")
    parser.add_argument("--metadata-csv", required=True, help="Path to DMS_substitutions.csv")
    parser.add_argument("--dms-dir", required=True, help="Path to DMS_ProteinGym_substitutions directory")
    parser.add_argument("--out", default="data/processed/proline_training_from_proteingym.csv")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Used only if DMS_score_bin is unavailable")
    parser.add_argument(
        "--structures-dir",
        default=None,
        help="Optional directory with PDB files. If provided, DMS positions are aligned and mapped to PDB residue numbering.",
    )
    parser.add_argument("--chain-id", default="A", help="Chain used for alignment mapping (default: A)")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    needed_meta_cols = {"DMS_id", "DMS_filename", "pdb_file"}
    missing = needed_meta_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    dms_dir = Path(args.dms_dir)
    rows = []
    missing_files = 0
    missing_pdb = 0
    missing_target_seq = 0
    unmapped_pos = 0
    mapped_pos = 0

    structures_dir = Path(args.structures_dir) if args.structures_dir else None
    pos_map_cache: dict[tuple[str, str], dict[int, int]] = {}

    for _, mrow in meta.iterrows():
        dms_file = dms_dir / str(mrow["DMS_filename"])
        if not dms_file.exists():
            missing_files += 1
            continue

        try:
            assay = pd.read_csv(dms_file)
        except Exception:
            continue

        if "mutant" not in assay.columns:
            continue

        has_bin = "DMS_score_bin" in assay.columns
        has_score = "DMS_score" in assay.columns
        if not has_bin and not has_score:
            continue

        protein_id = str(mrow["DMS_id"])
        pdb_file = str(mrow["pdb_file"])

        pos_map: dict[int, int] | None = None
        if structures_dir is not None:
            key = (protein_id, pdb_file)
            if key in pos_map_cache:
                pos_map = pos_map_cache[key]
            else:
                target_seq = str(mrow.get("target_seq", "")).strip().upper()
                if not target_seq:
                    missing_target_seq += 1
                    pos_map = {}
                else:
                    pdb_path = structures_dir / pdb_file
                    if not pdb_path.exists():
                        missing_pdb += 1
                        pos_map = {}
                    else:
                        try:
                            pdb_seq, pdb_resnums = extract_pdb_sequence_and_resnums(pdb_path, chain_id=args.chain_id)
                            pos_map = build_dms_to_pdb_pos_map(target_seq, pdb_seq, pdb_resnums)
                        except Exception:
                            pos_map = {}
                pos_map_cache[key] = pos_map

        for _, arow in assay.iterrows():
            parsed = parse_single_mutant(str(arow["mutant"]))
            if parsed is None:
                continue
            wt, pos, mut = parsed
            if mut != "P":
                continue

            if pos_map is not None:
                if pos not in pos_map:
                    unmapped_pos += 1
                    continue
                mapped_pos += 1
                pos = int(pos_map[pos])

            if has_bin:
                label = float(arow["DMS_score_bin"])
            else:
                label = 1.0 if float(arow["DMS_score"]) >= args.score_threshold else 0.0

            if label < 0.0 or label > 1.0:
                continue

            rows.append(
                {
                    "WT": wt,
                    "POS": pos,
                    "MUT": mut,
                    "LABEL": int(label),
                    "protein_id": protein_id,
                    "pdb_file": pdb_file,
                    "chain_id": "A",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No proline single-mutant rows produced from ProteinGym input")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out)} rows across {out['protein_id'].nunique()} proteins to {out_path}")
    print(f"Metadata entries with missing assay files: {missing_files}")
    if structures_dir is not None:
        total = mapped_pos + unmapped_pos
        mapping_rate = (mapped_pos / total) if total > 0 else float("nan")
        print(f"Mapping summary: mapped={mapped_pos} unmapped={unmapped_pos} rate={mapping_rate:.4f}")
        print(f"Metadata entries with missing target_seq: {missing_target_seq}")
        print(f"Metadata entries with missing pdb file: {missing_pdb}")


if __name__ == "__main__":
    main()
