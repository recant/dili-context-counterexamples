# Public pilot results

## Question

How much of a structure-only DILI classifier's apparent performance survives when the test chemistry is genuinely less familiar to the training set?

## Dataset and protocol

- Source filter: `DILIrank`
- Unique, non-conflicting structures after cleaning: **452**
- Positive fraction: **0.407**
- Repeated splits requested: **25** per split strategy
- Model: class-balanced logistic regression on radius-2, 2048-bit Morgan fingerprints
- Comparison: stratified random holdout vs Bemis-Murcko scaffold-group holdout
- Chemical novelty: maximum Morgan Tanimoto similarity from each test molecule to any training molecule

## Main result

Median random-split AUROC: **0.828**
Median scaffold-held-out AUROC: **0.801**
Random-to-scaffold AUROC gap: **0.027**

A positive gap means the random split makes the same structure-only model look better than it does when entire scaffolds are held out. That gap is an estimate of how much ordinary evaluation can benefit from chemical interpolation.

## Scaffold-held-out performance by chemical familiarity

| Max similarity to training | Median AUROC | Median AP | Median Brier | Median n/test split | Splits evaluable |
|---|---:|---:|---:|---:|---:|
| <0.30 | 0.738 | 0.744 | 0.215 | 45.000 | 25 |
| 0.30-0.50 | 0.845 | 0.764 | 0.166 | 28.000 | 25 |
| 0.50-0.70 | 1.000 | 1.000 | 0.050 | 11.000 | 25 |
| >=0.70 | 1.000 | 1.000 | 0.002 | 4.000 | 25 |

## Why this is useful for DILI-Context

The published DILI-Context work already shows that exposure-derived variables add signal beyond molecular structure in an aggregate benchmark. This stress test asks a different question: **does a model still work when chemistry is novel?**

The natural next experiment is to run the *same held-out compounds* with (1) structure only, (2) exposure/context only, and (3) structure + context. If context preferentially rescues the low-similarity bins, that is evidence that the model is using information beyond chemical interpolation. If it does not, that is an equally useful failure mode.

## Caveats

This is a small public baseline, not a reproduction of Absentia's internal model and not the full DILI-Context feature table. AUROC within narrow similarity bins can be undefined when a bin contains only one class; those cells are reported as NA. Repeated splits are not independent estimates of a deployment population.
