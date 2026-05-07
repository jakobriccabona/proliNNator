from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from transformers import AutoModel, AutoTokenizer

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


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


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def embed_sequence_chunked(
    seq: str,
    tokenizer,
    model,
    device: torch.device,
    max_residues: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    if len(seq) <= max_residues:
        toks = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks).last_hidden_state[0]
        return out[1 : 1 + len(seq)].detach().cpu().numpy().astype(np.float32)

    if overlap >= max_residues:
        raise ValueError("overlap must be < max_residues")

    step = max_residues - overlap
    chunks: list[tuple[int, int]] = []
    start = 0
    n = len(seq)
    while start < n:
        end = min(start + max_residues, n)
        chunks.append((start, end))
        if end == n:
            break
        start += step

    emb_sum = None
    emb_cnt = np.zeros((n, 1), dtype=np.float32)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        seq_batch = [seq[s:e] for s, e in batch]

        toks = tokenizer(seq_batch, return_tensors="pt", add_special_tokens=True, padding=True)
        toks = {k: v.to(device) for k, v in toks.items()}

        with torch.no_grad():
            out = model(**toks).last_hidden_state.detach().cpu().numpy().astype(np.float32)

        for b, (s, e) in enumerate(batch):
            L = e - s
            chunk_emb = out[b, 1 : 1 + L, :]
            if emb_sum is None:
                emb_sum = np.zeros((n, chunk_emb.shape[1]), dtype=np.float32)
            emb_sum[s:e] += chunk_emb
            emb_cnt[s:e] += 1.0

    emb_cnt[emb_cnt == 0.0] = 1.0
    assert emb_sum is not None
    return emb_sum / emb_cnt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-residue ESM2 embeddings for ProLinnator")
    parser.add_argument("--training-csv", required=True, help="CSV with protein_id and pdb_file columns")
    parser.add_argument("--structures-dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="HF model id or local directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npz embedding files")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.training_csv)
    required = {"protein_id", "pdb_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training csv missing columns: {sorted(missing)}")

    pairs = sorted({(str(r.protein_id), str(r.pdb_file)) for r in df[["protein_id", "pdb_file"]].itertuples(index=False)})

    device = choose_device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    model.to(device)

    success = 0
    skipped = 0

    for protein_id, pdb_file in pairs:
        out_file = out_dir / f"{protein_id}.npz"
        if out_file.exists():
            continue

        pdb_path = Path(args.structures_dir) / pdb_file
        if not pdb_path.exists():
            skipped += 1
            continue

        try:
            seq, resnums = extract_pdb_sequence_and_resnums(pdb_path, chain_id=args.chain_id)
            emb = embed_sequence_chunked(
                seq=seq,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_residues=args.max_residues,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )

            if args.dtype == "float16":
                emb_to_save = emb.astype(np.float16)
            else:
                emb_to_save = emb.astype(np.float32)

            np.savez_compressed(
                out_file,
                embeddings=emb_to_save,
                residue_numbers=np.array(resnums, dtype=np.int32),
            )
            success += 1
        except Exception as exc:
            print(f"[WARN] failed {protein_id} ({pdb_file}): {exc}")
            skipped += 1

    print(f"Done. success={success} skipped={skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
