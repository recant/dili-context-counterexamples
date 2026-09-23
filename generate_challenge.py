"""Build a small DILI scaffold-contradiction challenge set.

Expected input CSV columns:
    smiles,label,murcko_scaffold

The public pilot bundled with this repository was derived from the DILIrank-labelled
structure set released with StackDILI. This is a proof-of-concept benchmark, not
DILIrank 2.0 and not Absentia's DILI-Context dataset.
"""
from pathlib import Path
import argparse
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def build(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv).copy()
    required = {"smiles", "label", "murcko_scaffold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    if any(m is None for m in mols):
        raise ValueError("At least one SMILES failed RDKit parsing")
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(m) for m in mols]

    families, members = [], []
    fid = 0
    for scaffold, grp in df.groupby("murcko_scaffold"):
        if scaffold and grp.label.nunique() > 1 and len(grp) >= 2:
            fid += 1
            family_id = f"F{fid:02d}"
            families.append({
                "family_id": family_id,
                "murcko_scaffold": scaffold,
                "n_compounds": len(grp),
                "n_dili_positive": int((grp.label == 1).sum()),
                "n_dili_negative": int((grp.label == 0).sum()),
                "positive_fraction": float((grp.label == 1).mean()),
            })
            for idx, row in grp.iterrows():
                members.append({
                    "family_id": family_id,
                    "row_index": int(idx),
                    "label": int(row.label),
                    "smiles": row.smiles,
                    "murcko_scaffold": scaffold,
                })

    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.label.iloc[i] == df.label.iloc[j]:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            same_scaffold = (
                df.murcko_scaffold.iloc[i] != ""
                and df.murcko_scaffold.iloc[i] == df.murcko_scaffold.iloc[j]
            )
            pairs.append({
                "i": i,
                "j": j,
                "label_i": int(df.label.iloc[i]),
                "label_j": int(df.label.iloc[j]),
                "tanimoto": sim,
                "same_murcko_scaffold": same_scaffold,
                "scaffold": df.murcko_scaffold.iloc[i] if same_scaffold else "",
                "smiles_i": df.smiles.iloc[i],
                "smiles_j": df.smiles.iloc[j],
            })

    families_df = pd.DataFrame(families).sort_values("n_compounds", ascending=False)
    members_df = pd.DataFrame(members)
    pairs_df = pd.DataFrame(pairs).sort_values("tanimoto", ascending=False)

    families_df.to_csv(output_dir / "discordant_scaffold_families.csv", index=False)
    members_df.to_csv(output_dir / "challenge_set.csv", index=False)
    pairs_df.head(100).to_csv(output_dir / "top_opposite_label_analogs.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args()
    build(args.input_csv, args.output_dir)
