"""Stress-test DILI prediction under chemical novelty.

Question
--------
How much of a structure-only DILI classifier's apparent performance survives when
its test compounds are chemically novel rather than random held-out examples?

The script compares repeated random splits with Bemis-Murcko scaffold-held-out
splits, then stratifies test performance by each molecule's maximum Morgan-
fingerprint similarity to the training set. This makes chemical interpolation
visible instead of averaging it into a single AUROC.

By default the script downloads the public StackDILI dataset and restricts the
analysis to rows tagged DILIrank. You can instead pass a local CSV containing
SMILES and binary labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/GGCL7/StackDILI/main/Data/Dataset.csv"
)
SIMILARITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.000001]
SIMILARITY_LABELS = ["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    rename = {}
    if "smiles" not in out.columns:
        for candidate in ("canonical_smiles", "canonical smiles", "structure"):
            if candidate in out.columns:
                rename[candidate] = "smiles"
                break
    if "label" not in out.columns:
        for candidate in ("dili", "target", "y"):
            if candidate in out.columns:
                rename[candidate] = "label"
                break
    return out.rename(columns=rename)


def load_data(input_csv: str | None, source: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv or DEFAULT_DATA_URL)
    df = _normalise_columns(df)
    if not {"smiles", "label"}.issubset(df.columns):
        raise ValueError("Input must contain SMILES and Label columns")

    if source.lower() != "all" and "ref" in df.columns:
        mask = df["ref"].astype(str).str.lower() == source.lower()
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError(f"No rows matched source={source!r}")

    df = df[[c for c in df.columns if c in {"smiles", "label", "ref"}]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["smiles", "label"])
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    canonical, scaffolds, mols = [], [], []
    for s in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            canonical.append(None)
            scaffolds.append(None)
            mols.append(None)
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        # Empty Murcko scaffolds (usually acyclic molecules) should not all be
        # forced into one giant group. Treat each canonical acyclic structure
        # as its own scaffold group.
        if not scaffold:
            scaffold = f"ACYCLIC::{can}"
        canonical.append(can)
        scaffolds.append(scaffold)
        mols.append(mol)

    df["canonical_smiles"] = canonical
    df["scaffold"] = scaffolds
    df["mol"] = mols
    df = df.dropna(subset=["canonical_smiles", "scaffold", "mol"]).copy()

    # Remove structures with conflicting labels, then deduplicate exact
    # structures so identical chemistry cannot leak across train/test.
    nlabels = df.groupby("canonical_smiles")["label"].nunique()
    conflicting = set(nlabels[nlabels > 1].index)
    if conflicting:
        df = df[~df["canonical_smiles"].isin(conflicting)].copy()
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Need both positive and negative labels after cleaning")
    return df


def fingerprints(mols: list[Chem.Mol], radius: int = 2, n_bits: int = 2048):
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [generator.GetFingerprint(m) for m in mols]
    x = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, x[i])
    return fps, x


def safe_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_fraction": float(np.mean(y_true)) if len(y_true) else np.nan,
        "brier": float(brier_score_loss(y_true, p)) if len(y_true) else np.nan,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, pred)) if len(y_true) else np.nan
        ),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["average_precision"] = float(average_precision_score(y_true, p))
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
    return out


def max_train_similarity(test_indices, train_indices, fps) -> np.ndarray:
    train_fps = [fps[i] for i in train_indices]
    values = []
    for i in test_indices:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        values.append(max(sims) if sims else np.nan)
    return np.asarray(values, dtype=float)


def valid_split(y: np.ndarray, train_idx, test_idx) -> bool:
    return len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2


def evaluate_split(
    split_type: str,
    split_id: int,
    train_idx,
    test_idx,
    y: np.ndarray,
    x: np.ndarray,
    fps,
    df: pd.DataFrame,
):
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=split_id,
    )
    model.fit(x[train_idx], y[train_idx])
    p = model.predict_proba(x[test_idx])[:, 1]
    similarity = max_train_similarity(test_idx, train_idx, fps)

    overall = {
        "split_type": split_type,
        "split_id": split_id,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "mean_max_train_similarity": float(np.mean(similarity)),
        **safe_metrics(y[test_idx], p),
    }

    pred_rows = pd.DataFrame(
        {
            "split_type": split_type,
            "split_id": split_id,
            "row_index": test_idx,
            "canonical_smiles": df.iloc[test_idx]["canonical_smiles"].to_numpy(),
            "label": y[test_idx],
            "probability": p,
            "max_train_similarity": similarity,
        }
    )
    pred_rows["similarity_bin"] = pd.cut(
        pred_rows["max_train_similarity"],
        bins=SIMILARITY_BINS,
        labels=SIMILARITY_LABELS,
        right=False,
        include_lowest=True,
    )

    bin_rows = []
    for label in SIMILARITY_LABELS:
        subset = pred_rows[pred_rows["similarity_bin"] == label]
        if subset.empty:
            continue
        metrics = safe_metrics(
            subset["label"].to_numpy(), subset["probability"].to_numpy()
        )
        bin_rows.append(
            {
                "split_type": split_type,
                "split_id": split_id,
                "similarity_bin": label,
                **metrics,
            }
        )
    return overall, bin_rows, pred_rows


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def write_summary(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    output_dir: Path,
    source: str,
    repeats: int,
) -> None:
    rows = []
    for split_type in ("random", "scaffold"):
        g = metrics[metrics["split_type"] == split_type]
        rows.append(
            {
                "split": split_type,
                "median_auc": g["roc_auc"].median(),
                "median_ap": g["average_precision"].median(),
                "median_brier": g["brier"].median(),
                "median_similarity": g["mean_max_train_similarity"].median(),
                "n_splits": len(g),
            }
        )
    summary = pd.DataFrame(rows).set_index("split")
    random_auc = summary.loc["random", "median_auc"] if "random" in summary.index else np.nan
    scaffold_auc = summary.loc["scaffold", "median_auc"] if "scaffold" in summary.index else np.nan
    gap = random_auc - scaffold_auc

    scaffold_bins = bins[bins["split_type"] == "scaffold"]
    bin_summary = (
        scaffold_bins.groupby("similarity_bin", observed=True)
        .agg(
            median_auc=("roc_auc", "median"),
            median_ap=("average_precision", "median"),
            median_brier=("brier", "median"),
            median_n=("n", "median"),
            evaluated_splits=("split_id", "nunique"),
        )
        .reindex(SIMILARITY_LABELS)
    )

    lines = [
        "# Public pilot results",
        "",
        "## Question",
        "",
        "How much of a structure-only DILI classifier's apparent performance survives when the test chemistry is genuinely less familiar to the training set?",
        "",
        "## Dataset and protocol",
        "",
        f"- Source filter: `{source}`",
        f"- Unique, non-conflicting structures after cleaning: **{len(df)}**",
        f"- Positive fraction: **{df['label'].mean():.3f}**",
        f"- Repeated splits requested: **{repeats}** per split strategy",
        "- Model: class-balanced logistic regression on radius-2, 2048-bit Morgan fingerprints",
        "- Comparison: stratified random holdout vs Bemis-Murcko scaffold-group holdout",
        "- Chemical novelty: maximum Morgan Tanimoto similarity from each test molecule to any training molecule",
        "",
        "## Main result",
        "",
        f"Median random-split AUROC: **{_fmt(random_auc)}**",
        f"Median scaffold-held-out AUROC: **{_fmt(scaffold_auc)}**",
        f"Random-to-scaffold AUROC gap: **{_fmt(gap)}**",
        "",
        "A positive gap means the random split makes the same structure-only model look better than it does when entire scaffolds are held out. That gap is an estimate of how much ordinary evaluation can benefit from chemical interpolation.",
        "",
        "## Scaffold-held-out performance by chemical familiarity",
        "",
        "| Max similarity to training | Median AUROC | Median AP | Median Brier | Median n/test split | Splits evaluable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in bin_summary.iterrows():
        lines.append(
            f"| {label} | {_fmt(row['median_auc'])} | {_fmt(row['median_ap'])} | {_fmt(row['median_brier'])} | {_fmt(row['median_n'])} | {int(row['evaluated_splits']) if pd.notna(row['evaluated_splits']) else 0} |"
        )

    lines += [
        "",
        "## Why this is useful for DILI-Context",
        "",
        "The published DILI-Context work already shows that exposure-derived variables add signal beyond molecular structure in an aggregate benchmark. This stress test asks a different question: **does a model still work when chemistry is novel?**",
        "",
        "The natural next experiment is to run the *same held-out compounds* with (1) structure only, (2) exposure/context only, and (3) structure + context. If context preferentially rescues the low-similarity bins, that is evidence that the model is using information beyond chemical interpolation. If it does not, that is an equally useful failure mode.",
        "",
        "## Caveats",
        "",
        "This is a small public baseline, not a reproduction of Absentia's internal model and not the full DILI-Context feature table. AUROC within narrow similarity bins can be undefined when a bin contains only one class; those cells are reported as NA. Repeated splits are not independent estimates of a deployment population.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_csv: str | None, output_dir: Path, source: str, repeats: int, test_size: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_csv, source)
    fps, x = fingerprints(df["mol"].tolist())
    y = df["label"].to_numpy(dtype=int)
    indices = np.arange(len(df))

    metric_rows = []
    bin_rows = []
    prediction_frames = []

    for seed in range(repeats):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
        overall, bins, preds = evaluate_split(
            "random", seed, train_idx, test_idx, y, x, fps, df
        )
        metric_rows.append(overall)
        bin_rows.extend(bins)
        prediction_frames.append(preds)

    groups = df["scaffold"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=repeats,
        test_size=test_size,
        random_state=0,
    )
    accepted = 0
    for candidate_id, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups)):
        if not valid_split(y, train_idx, test_idx):
            continue
        overall, bins, preds = evaluate_split(
            "scaffold", accepted, train_idx, test_idx, y, x, fps, df
        )
        metric_rows.append(overall)
        bin_rows.extend(bins)
        prediction_frames.append(preds)
        accepted += 1

    if accepted == 0:
        raise RuntimeError("No scaffold split contained both classes in train and test")

    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    clean = df.drop(columns=["mol"]).copy()
    clean.to_csv(output_dir / "clean_dataset.csv", index=False)
    metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    bins.to_csv(output_dir / "similarity_bin_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_summary(df, metrics, bins, output_dir, source, repeats)

    print((output_dir / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Local/remote CSV. Defaults to the public StackDILI dataset.",
    )
    parser.add_argument(
        "--source",
        default="DILIrank",
        help="If the CSV has a ref column, keep this source (default: DILIrank). Use 'all' to disable filtering.",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.input_csv, args.output_dir, args.source, args.repeats, args.test_size)
