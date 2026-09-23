# DILI chemical generalization stress test

## Question

**How much of a structure-only DILI classifier's apparent performance survives when the test compounds are chemically novel rather than random held-out examples?**

This is deliberately different from asking whether similar scaffolds can carry different DILI labels. DILI-Context already establishes that exposure matters and reports complementary signal from exposure-derived covariates. The more useful follow-up is whether apparent predictive performance survives when a model cannot lean as heavily on familiar chemistry.

That lines up directly with Absentia's public research question: how do we tell genuine generalization on a novel compound from interpolation over known chemistry?

## Public result

Using the public DILIrank-labelled structures distributed with StackDILI, this repo currently finds:

- **452** unique, non-conflicting structures after cleaning
- median random-split AUROC: **0.828**
- median Bemis-Murcko scaffold-held-out AUROC: **0.801**
- random-to-scaffold AUROC gap: **0.027**
- among scaffold-held-out compounds with maximum Morgan Tanimoto similarity **< 0.30** to any training molecule, median AUROC falls to **0.738** and median Brier score is **0.215**

The high-similarity bins are much easier, although they are also small, so they should not be overinterpreted. Full split-level and prediction-level outputs are in [`results/`](results/).

See [`results/SUMMARY.md`](results/SUMMARY.md) for the generated summary.

## What the benchmark does

For the same class-balanced logistic-regression baseline on radius-2, 2048-bit Morgan fingerprints, it compares:

1. repeated stratified random holdouts;
2. repeated Bemis-Murcko scaffold-group holdouts;
3. performance and calibration as a function of each test molecule's maximum fingerprint similarity to the training set.

The point is not to claim this simple model is state of the art. The point is to expose when evaluation is benefiting from chemical familiarity.

## The DILI-Context experiment this sets up

The published DILI-Context work already reports that exposure-derived variables add signal beyond structure in an aggregate benchmark. The next experiment is therefore much sharper:

> **When chemistry is genuinely unfamiliar, do dose/exposure/context features rescue the predictions that a structure-only model gets wrong?**

Run the exact same held-out compounds through:

1. structure only;
2. context only;
3. structure + context.

Then compare the gain from context across chemical-similarity bins. If context helps disproportionately in the low-similarity bins, that is evidence that it is contributing information beyond chemical interpolation. If it does not, that is a useful failure mode too.

This repo does **not** currently claim to answer that second question because it does not bundle Absentia's full DILI-Context feature table.

## Reproduce

```bash
pip install -r requirements.txt
python benchmark_generalization.py --output-dir results
```

By default the script downloads the public StackDILI dataset and filters to rows tagged `DILIrank`.

You can run another compatible binary DILI dataset with:

```bash
python benchmark_generalization.py --input-csv your_data.csv --source all
```

The CSV must contain `SMILES` and `Label` columns (case-insensitive), with binary labels 0/1.

## Outputs

- `results/SUMMARY.md` — generated headline results and caveats
- `results/split_metrics.csv` — metrics for every random/scaffold split
- `results/similarity_bin_metrics.csv` — split-level metrics by chemical-familiarity bin
- `results/predictions.csv` — held-out prediction, label, and max training similarity for every evaluated compound
- `results/clean_dataset.csv` — cleaned structures used by the benchmark

A GitHub Action reruns the benchmark whenever the benchmark code or dependencies change and commits the generated outputs.

## Public provenance

- DILI-Context overview: https://www.absentia.bio/publications/dili-context
- Absentia AI Research Scientist research questions: https://jobs.ashbyhq.com/absentia-labs/f8cb711d-2f7e-457a-856e-c455fa541ddc
- StackDILI public dataset: https://github.com/GGCL7/StackDILI/tree/main/Data

## Caveats

This is a small public baseline, not a reproduction of Absentia's internal model. Repeated train/test splits are not independent deployment cohorts. Similarity-stratified AUROC can also become unstable when a bin is small or contains only one class; undefined values are left as `NA` rather than silently filled.
