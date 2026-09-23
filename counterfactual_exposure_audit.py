"""Generate and audit counterfactual exposure challenges for DILI models.

Two challenge families are supported:

1. dose_ladder: keep a drug/context fixed and vary daily dose over a log-spaced
   multiplier grid.
2. ablation: keep the reference row fixed and remove one context field at a time.

The script is model-agnostic. Generate challenge rows, score them with any model,
then add a `predicted_risk` column (and optionally `uncertainty`) and run `audit`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

DEFAULT_MULTIPLIERS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_ABLATION_FIELDS = ["daily_dose_mg", "duration_days", "noael_mg_kg_day", "route"]


def _require(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_challenges(
    input_csv: Path,
    output_csv: Path,
    multipliers: list[float],
    ablation_fields: list[str],
) -> None:
    df = pd.read_csv(input_csv)
    _require(df, ["drug_id", "daily_dose_mg"])
    df = df.copy()
    df["daily_dose_mg"] = pd.to_numeric(df["daily_dose_mg"], errors="coerce")
    df = df.dropna(subset=["drug_id", "daily_dose_mg"])
    df = df[df["daily_dose_mg"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with positive numeric daily_dose_mg")

    rows: list[dict] = []
    for idx, row in df.iterrows():
        base = row.to_dict()
        context_id = str(base.get("context_id", f"ctx_{idx:05d}"))
        base_dose = float(base["daily_dose_mg"])

        for m in multipliers:
            r = dict(base)
            r.update(
                {
                    "context_id": context_id,
                    "challenge_type": "dose_ladder",
                    "challenge_field": "daily_dose_mg",
                    "challenge_value": base_dose * float(m),
                    "challenge_multiplier": float(m),
                    "is_reference": bool(math.isclose(float(m), 1.0)),
                    "base_daily_dose_mg": base_dose,
                }
            )
            r["daily_dose_mg"] = base_dose * float(m)
            rows.append(r)

        # Ablations are separate from dose ladders. Missing numeric values become
        # NaN and text values become an empty string so the downstream model can
        # apply its own missing-data policy.
        for field in ablation_fields:
            if field not in base:
                continue
            r = dict(base)
            r.update(
                {
                    "context_id": context_id,
                    "challenge_type": "ablation",
                    "challenge_field": field,
                    "challenge_value": np.nan,
                    "challenge_multiplier": np.nan,
                    "is_reference": False,
                    "base_daily_dose_mg": base_dose,
                }
            )
            r[field] = np.nan if pd.api.types.is_number(base[field]) else ""
            rows.append(r)

        # Include one untouched reference row for comparing ablation uncertainty.
        ref = dict(base)
        ref.update(
            {
                "context_id": context_id,
                "challenge_type": "reference",
                "challenge_field": "none",
                "challenge_value": base_dose,
                "challenge_multiplier": 1.0,
                "is_reference": True,
                "base_daily_dose_mg": base_dose,
            }
        )
        rows.append(ref)

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out):,} challenge rows to {output_csv}")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.rank().corr(y.rank()))


def audit_predictions(input_csv: Path, output_dir: Path, epsilon: float) -> None:
    df = pd.read_csv(input_csv)
    _require(
        df,
        [
            "drug_id",
            "context_id",
            "challenge_type",
            "challenge_field",
            "predicted_risk",
        ],
    )
    df["predicted_risk"] = pd.to_numeric(df["predicted_risk"], errors="coerce")
    df = df.dropna(subset=["predicted_risk"]).copy()
    if df.empty:
        raise ValueError("No numeric predicted_risk values found")

    ladders = df[df["challenge_type"] == "dose_ladder"].copy()
    _require(ladders, ["daily_dose_mg", "challenge_multiplier"])
    ladders["daily_dose_mg"] = pd.to_numeric(ladders["daily_dose_mg"], errors="coerce")
    ladders = ladders.dropna(subset=["daily_dose_mg"])

    per_context = []
    reversals = []
    for (drug_id, context_id), g in ladders.groupby(["drug_id", "context_id"], dropna=False):
        g = g.sort_values("daily_dose_mg").copy()
        risk = g["predicted_risk"].to_numpy(float)
        dose = g["daily_dose_mg"].to_numpy(float)
        if len(g) < 2:
            continue
        diffs = np.diff(risk)
        step_reversal = diffs < -epsilon
        for i, flag in enumerate(step_reversal):
            if flag:
                reversals.append(
                    {
                        "drug_id": drug_id,
                        "context_id": context_id,
                        "dose_low": dose[i],
                        "dose_high": dose[i + 1],
                        "risk_low": risk[i],
                        "risk_high": risk[i + 1],
                        "risk_change": diffs[i],
                    }
                )
        per_context.append(
            {
                "drug_id": drug_id,
                "context_id": context_id,
                "n_doses": len(g),
                "dose_min": float(np.min(dose)),
                "dose_max": float(np.max(dose)),
                "risk_min": float(np.min(risk)),
                "risk_max": float(np.max(risk)),
                "risk_range": float(np.max(risk) - np.min(risk)),
                "spearman_logdose_risk": _spearman(pd.Series(np.log(dose)), pd.Series(risk)),
                "n_adjacent_reversals": int(step_reversal.sum()),
                "adjacent_reversal_fraction": float(step_reversal.mean()),
                "largest_negative_step": float(min(0.0, np.min(diffs))),
            }
        )

    context_metrics = pd.DataFrame(per_context)
    reversal_df = pd.DataFrame(reversals)

    summary_lines = [
        "# Counterfactual exposure audit",
        "",
        "## Dose-ladder results",
        "",
    ]
    if context_metrics.empty:
        summary_lines.append("No complete dose ladders were available for scoring.")
    else:
        n_contexts = len(context_metrics)
        with_reversal = int((context_metrics["n_adjacent_reversals"] > 0).sum())
        total_steps = int((context_metrics["n_doses"] - 1).sum())
        total_reversals = int(context_metrics["n_adjacent_reversals"].sum())
        summary_lines += [
            f"- Contexts audited: **{n_contexts}**",
            f"- Contexts with >=1 dose reversal (> {epsilon:g} risk units): **{with_reversal} / {n_contexts} ({with_reversal / n_contexts:.1%})**",
            f"- Adjacent dose steps that reverse: **{total_reversals} / {total_steps} ({total_reversals / total_steps:.1%})**" if total_steps else "- No adjacent dose steps available.",
            f"- Median within-context Spearman(log-dose, risk): **{context_metrics['spearman_logdose_risk'].median():.3f}**",
            f"- Median risk range across the 64x dose ladder: **{context_metrics['risk_range'].median():.3f}**",
            "",
            "A reversal means predicted risk decreased when dose increased while the molecule and all other supplied context stayed fixed. This is a challenge flag, not automatic proof of biological error.",
        ]

    # Optional uncertainty-ablation audit.
    if "uncertainty" in df.columns:
        u = df.copy()
        u["uncertainty"] = pd.to_numeric(u["uncertainty"], errors="coerce")
        refs = (
            u[u["challenge_type"] == "reference"]
            .dropna(subset=["uncertainty"])
            .groupby(["drug_id", "context_id"], as_index=False)["uncertainty"]
            .mean()
            .rename(columns={"uncertainty": "reference_uncertainty"})
        )
        ab = u[u["challenge_type"] == "ablation"].dropna(subset=["uncertainty"]).copy()
        ab = ab.merge(refs, on=["drug_id", "context_id"], how="left")
        ab["uncertainty_change"] = ab["uncertainty"] - ab["reference_uncertainty"]
        ab.to_csv(output_dir / "ablation_uncertainty.csv", index=False)
        summary_lines += ["", "## Context-ablation uncertainty", ""]
        if ab.empty:
            summary_lines.append("No scored ablation rows with uncertainty were available.")
        else:
            by_field = ab.groupby("challenge_field")["uncertainty_change"].agg(["count", "median", "mean"])
            summary_lines += [
                "Positive values mean uncertainty increased after the field was removed.",
                "",
                "| Removed field | n | Median uncertainty change | Mean uncertainty change |",
                "|---|---:|---:|---:|",
            ]
            for field, row in by_field.iterrows():
                summary_lines.append(
                    f"| {field} | {int(row['count'])} | {row['median']:.4f} | {row['mean']:.4f} |"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    context_metrics.to_csv(output_dir / "dose_context_metrics.csv", index=False)
    reversal_df.to_csv(output_dir / "dose_reversals.csv", index=False)
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print((output_dir / "SUMMARY.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make_p = sub.add_parser("make", help="Generate counterfactual challenge rows")
    make_p.add_argument("--input", required=True, type=Path)
    make_p.add_argument("--output", required=True, type=Path)
    make_p.add_argument(
        "--multipliers",
        default=",".join(str(x) for x in DEFAULT_MULTIPLIERS),
        help="Comma-separated dose multipliers",
    )
    make_p.add_argument(
        "--ablate",
        default=",".join(DEFAULT_ABLATION_FIELDS),
        help="Comma-separated fields to ablate when present",
    )

    audit_p = sub.add_parser("audit", help="Audit model-scored challenge rows")
    audit_p.add_argument("--input", required=True, type=Path)
    audit_p.add_argument("--output-dir", required=True, type=Path)
    audit_p.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Minimum risk decrease counted as a reversal (default: 0.01)",
    )

    args = parser.parse_args()
    if args.command == "make":
        multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
        ablations = [x.strip() for x in args.ablate.split(",") if x.strip()]
        make_challenges(args.input, args.output, multipliers, ablations)
    else:
        audit_predictions(args.input, args.output_dir, args.epsilon)


if __name__ == "__main__":
    main()
