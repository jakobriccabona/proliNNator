from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create list of structure files referenced in DMS_substitutions.csv")
    parser.add_argument("--dms-csv", required=True)
    parser.add_argument("--out", default="data/processed/required_pdb_files.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.dms_csv)
    if "pdb_file" not in df.columns:
        raise ValueError("Expected 'pdb_file' column in DMS metadata CSV")

    pdbs = sorted({str(x).strip() for x in df["pdb_file"].dropna().tolist() if str(x).strip()})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(pdbs) + "\n", encoding="utf-8")
    print(f"Wrote {len(pdbs)} structure filenames to {out_path}")


if __name__ == "__main__":
    main()
