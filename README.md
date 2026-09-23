# DILI Context Counterexamples

A small proof-of-concept benchmark for a specific question: **when chemical structure gives the wrong intuition about DILI risk, can dose, exposure, and biological context resolve the counterexample?**

## Motivation

Absentia's DILI-Context work argues that clinical hepatotoxicity is exposure-dependent rather than a fixed property of molecular structure. Their public research agenda also emphasizes out-of-distribution generalization, applicability-domain assessment, mechanistic interpretation, and benchmarks that distinguish genuine biological generalization from chemical interpolation.

This repository turns that into an adversarial evaluation design. It identifies compounds that share the same Bemis-Murcko scaffold, or are close structural analogs, but carry discordant DILI labels. These are cases where a structure-only model cannot safely rely on scaffold identity. A context-aware model should have an opportunity to explain the divergence using dose, duration, exposure, toxicity thresholds, metabolism, or biological evidence.

## Public pilot

The bundled pilot uses a small public, structure-resolved DILIrank-labelled set released with StackDILI. It is **not** the full FDA DILIrank 2.0 dataset and is **not** Absentia's DILI-Context dataset.

Pilot observations:

- 4 exact Murcko-scaffold families contain both DILI-positive and DILI-negative examples.
- 21 compounds fall into those discordant families.
- The largest family contains 11 compounds: 4 positive and 7 negative.
- The closest opposite-label Morgan-fingerprint pair in this pilot has Tanimoto 0.481.

The last point is important: the pilot establishes the benchmark construction, but the chemistry is not yet close enough to support strong claims about "chemical twins." The useful next step is to run the pipeline on full DILIrank 2.0 / DILI-Context and mine genuinely high-similarity discordant pairs.

## Full benchmark design

1. Standardize and map the full DILIrank 2.0 compound set to parent structures.
2. Mine exact-scaffold and high-Tanimoto pairs with discordant concern/severity labels.
3. Attach context variables such as therapeutic dose, duration, route, NOAEL/LOAEL, safety margins, hepatic-adjustment evidence, metabolism, and targets.
4. Compare structure-only predictions with progressively context-enriched models.
5. Evaluate **counterexample resolution rate**: among structurally misleading pairs, how often does context recover the correct risk ordering?
6. Require attribution: which context variable(s) caused the risk ordering to change?
7. Hold out entire contradiction families to reduce scaffold memorization and leakage.

The intended endpoint is not another aggregate AUROC. It is a falsifiable test of whether a model can explain clinically divergent outcomes among structurally related drugs.

## Files

- `challenge_set.csv` — compounds belonging to exact scaffold families with discordant labels.
- `discordant_scaffold_families.csv` — family-level summary.
- `top_opposite_label_analogs.csv` — highest-similarity opposite-label pairs in the pilot.
- `nearest_neighbor_audit.csv` — nearest structural neighbor and label agreement for every pilot compound.
- `similarity_stratified_contradictions.csv` — contradiction rates stratified by nearest-neighbor similarity.
- `generate_challenge.py` — minimal reproducible construction script.

## Public provenance

- DILI-Context overview: https://www.absentia.bio/publications/dili-context
- Absentia AI Research Scientist research questions: https://jobs.ashbyhq.com/absentia-labs/f8cb711d-2f7e-457a-856e-c455fa541ddc
- Public structure-resolved pilot source: https://github.com/GGCL7/StackDILI

## Status

Proof of concept. The repository intentionally separates demonstrated pilot results from the proposed full DILIrank 2.0 / DILI-Context experiment.
