# DILI counterfactual exposure audit

## The question

**If a DILI model says it uses dose and exposure, does it actually behave sensibly when you change dose for the same molecule?**

DILI-Context already shows that dose/exposure variables are associated with DILI severity across drugs and that exposure-enriched features add predictive signal. This repo asks a different question: whether an exposure-aware model is *counterfactually consistent* rather than merely using dose as a useful correlational feature across the dataset.

For one drug, freeze the chemistry and every other context variable, then change only the dose. If the model's predicted DILI risk falls sharply as dose rises, that is a counterfactual reversal worth investigating. Likewise, if removing an exposure variable makes the model *more* confident, that is a useful uncertainty failure case.

I could not find either evaluation reported in the public DILI-Context paper, Absentia's public writeup, or its current research-problem description. This does **not** establish that Absentia has never run these tests internally.

## Challenge 1: same-drug dose ladders

For each drug/context row, generate counterfactual copies at:

`0.125x, 0.25x, 0.5x, 1x, 2x, 4x, 8x` the reference daily dose.

Everything except dose is held fixed.

After the model scores the rows, the audit reports:

- fraction of adjacent dose steps where predicted risk decreases;
- fraction of drugs with at least one reversal;
- largest reversal for each drug;
- Spearman correlation between log-dose and predicted risk within each drug;
- dose-response range, to identify models that claim to use exposure but are effectively flat.

This is intentionally a **behavioral sanity test**, not a claim that every real DILI mechanism is globally monotonic over arbitrary doses. Large local reversals are flags for inspection, not automatic proof that a model is biologically wrong.

## Challenge 2: exposure-information ablation

For each reference context, make copies with individual context fields removed (for example dose, duration, NOAEL, or route). If a model emits uncertainty/confidence, the audit can test whether uncertainty rises when information it supposedly uses disappears.

This is useful for separating:

- a model that actually depends on exposure context;
- a model that mostly ignores it;
- a model that depends on it but is overconfident when it is missing.

## Why this is different from the published DILI-Context benchmark

The published work asks whether exposure-derived variables stratify DILI concern and add predictive signal across a cohort. This benchmark asks a within-molecule intervention question:

> **Holding the drug fixed, does changing exposure move the model in the expected direction?**

A model can perform well on an ordinary exposure-enriched classification benchmark and still fail this test.

## Usage

Start with a CSV containing at minimum:

```text
drug_id,daily_dose_mg
```

You can include any other model inputs (SMILES, route, duration, NOAEL, targets, etc.); they are carried through unchanged.

Generate counterfactuals:

```bash
python counterfactual_exposure_audit.py make \
  --input contexts.csv \
  --output counterfactuals.csv
```

Run your model on `counterfactuals.csv` and add a `predicted_risk` column. If available, also add `uncertainty`.

Audit predictions:

```bash
python counterfactual_exposure_audit.py audit \
  --input scored_counterfactuals.csv \
  --output-dir audit_results
```

The script writes per-drug metrics, individual reversals, and a short Markdown summary.

## Required columns for scoring

The generator adds the metadata needed by the audit automatically. The scored file needs:

- `predicted_risk` — larger means more predicted DILI risk;
- optionally `uncertainty` — larger means less confidence.

No Absentia model or proprietary data is included here. The point is to provide a small, falsifiable evaluation that can be run against any exposure-aware DILI model.

## Public provenance

- DILI-Context paper: https://doi.org/10.1093/toxsci/kfag077
- Absentia DILI-Context writeup: https://www.absentia.bio/publications/dili-context
- Absentia AI research questions: https://jobs.ashbyhq.com/absentia-labs/f8cb711d-2f7e-457a-856e-c455fa541ddc

## Caveat

Dose-response relationships can be nonlinear and some idiosyncratic mechanisms need more context than dose alone. The benchmark therefore treats monotonicity violations as **challenge cases to inspect**, not as definitive biological errors. Its purpose is to expose model behavior that aggregate AUROC cannot show.
