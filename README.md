# CABBAGE-Bench

Repository: <https://github.com/Nesma876/Cabbage-Bench>

An ML-ready, multi-label, shift-aware benchmark for genotype-to-phenotype
antimicrobial resistance prediction, built on the CABBAGE database
(EMBL-EBI AMR Portal, release `2025-11`).

Six priority species, 61 247 isolates, four evaluation protocols
(random, lineage-aware, geography-aware, time-aware), three genomic
representations, and four baseline families including a rule-based
comparator.

The six species are *Acinetobacter baumannii*, *Escherichia coli*,
*Klebsiella pneumoniae*, *Pseudomonas aeruginosa*, *Salmonella enterica* and
*Staphylococcus aureus*.

This repository holds the code and the result files. The derived data —
label matrices, feature matrices and split indices — are deposited
separately on Zenodo (see **Data** below), because they are too large for
version control.

---

## What is here

```
pipeline/     building the benchmark, from the CABBAGE parquet files
  cellule_tout.py        labels, feature matrices, lineages, splits
  run_all.py             driver over the six species
  deduplication.py       near-clonal twin audit
  refaire_matrice.py     gene-content matrix (annotated gene names)
  refaire_lignees.py     accessory-content lineage clustering
  baselines_forts.py     rules, logistic, random forest, gradient boosting
  geographie.py          leave-one-region-out with size-matched controls
  decomposition.py       crossed geography x provenance holdout
  robustesse.py          ten-seed split stability
  stats_benchmark.py     descriptive statistics
  figures.py             figures 1, 2, 4 and the appendix figures

analyse/      the statistics reported in the paper
  tableau7.py            rule margins, three representations
  marges.py              per-protocol margin collapse
  decomp_apparie.py      paired decomposition, clustered bootstrap
  lignee_controle.py     lineage-novelty control, three representations
  fig3_confusions.py     figure 3

jeanzay/      running the chain on an HPC cluster without compute-node network
  18a_preparer.py        stage the parquet files and the annotation cache
  18b_normes_calcul.slurm  single-standard reruns (CLSI / EUCAST)

resultats/    the result files every number in the paper is computed from
```

## The three feature representations

Selected by the environment variable `CABBAGE_MATRICE`:

| value | vocabulary | source |
|---|---|---|
| `amr` | resistance determinants | AMRFinderPlus `amr_element_symbol` |
| `pan` | annotated gene names | per-assembly GFF, paralogue suffixes merged |
| `pg` | orthologue families | MMseqs2 `easy-linclust`, 90 % identity, 80 % coverage |

The resistance-determinant block is retained unconditionally in all three, so
the rule-based comparator is numerically identical across them. We verify
this: its AUROC and very-major-error rate agree to the last stored digit on
all 526 shared (species, protocol, drug) cells.

## Two conventions that matter

Both are stated here because results in the paper are sensitive to them.

1. A (species, drug) cell enters an analysis only if its test set contains at
   least **30 isolates**.
2. Where a condition is compared against several random control draws, the
   cell's control value is the **mean** over those draws.

Missing phenotypes are represented explicitly by `M.npy`. A zero in the mask
means “not observed” and must never be interpreted as a susceptible label;
metrics and losses are computed only where the mask is one.

Each species directory in the data archive contains 13 files: `Y.npy` and
`M.npy`, antimicrobial names, isolate metadata, lineage assignments, three
sparse feature matrices and their vocabularies, fixed split assignments and a
machine-readable summary. See `DATA_PROVENANCE.md` for the derivation and
`DATA_MANIFEST.tsv` for byte sizes and checksums.

## Phenotype standard

The default label is `coalesce(Updated_phenotype_EUCAST,
Updated_phenotype_CLSI, resistance_phenotype)`, chosen per isolate by
availability. Because availability varies by region, this mixes breakpoint
conventions within a single label set. Setting `CABBAGE_NORME=clsi` or
`=eucast` restricts labels to one convention, discarding pairs with no value
in that column rather than filling them from the other.

## Reproducing

```bash
pip install -r requirements.txt
export CABBAGE_MATRICE=amr
python pipeline/run_all.py --sortie bench --especes "Klebsiella pneumoniae" ...
python pipeline/baselines_forts.py --bench bench
python pipeline/geographie.py      --bench bench
python analyse/tableau7.py
```

The scripts in `analyse/` regenerate the statistical tables from the archived
CSV files. Newly generated outputs should be written outside `resultats/`
until they have been compared with the archived tables.

On a cluster whose compute nodes have no outbound network, run
`jeanzay/18a_preparer.py` on a login node first; it stages the parquet files
and the annotation cache to disk, after which `CABBAGE_PARQUET` makes the
whole chain run offline.

## Data

Label matrices, feature matrices and split indices are prepared for a single
Zenodo record: <https://doi.org/10.5281/zenodo.23732198>. Version 1.0.0
contains exactly `cabbage-bench-depot.zip` (code, documentation and archived
result tables) and `cabbage-bench-donnees.tgz` (derived benchmark data). The
data archive is 73,396,037 bytes and has SHA-256
`57e2f5f70326a044c266852ed546672936b0d37ae6b9e83fb4f50ecfb9348171`.

To validate an extracted data archive and the checked-in result tables:

```bash
python scripts/verify_release.py --data /path/to/bench --results resultats
```

The validator checks array dimensions, masks, sparse-matrix row alignment,
the four fixed protocol assignments, cohort totals, and the principal
published aggregates.

Upstream source: the CABBAGE database, EMBL-EBI AMR Portal, release
`2025-11`, <https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases/>.
CABBAGE is redistributed under its own terms; this repository adds the
benchmark construction, the splits and the analysis.

## Citing

Hamza Cherif S., Settouti N. (2026). *CABBAGE-Bench: an ML-ready,
shift-aware benchmark for genotype-to-phenotype antimicrobial resistance
prediction* (Version 1.0.0) [Dataset]. Zenodo.
<https://doi.org/10.5281/zenodo.23732198>.

## Licence

Code: MIT. Derived data: CC BY 4.0. The CABBAGE article states that portal data
and genome annotations are distributed under CC BY 4.0, and the benchmark
authors confirm that all 78 released files derive exclusively from that
redistributable material. Attribution and transformation details are recorded
in `LICENSE-DATA.md`, `DATA_PROVENANCE.md` and
`LICENCE_AND_PROVENANCE_AUDIT.md`.

## Verified release results

The release validator confirms 61 247 isolates, 58 861 annotated genomes, 424
descriptive region--drug comparisons, 20/21 degrading regions, a median
geographic AUROC change of -0.09324 and a region-level Wilcoxon p-value of
2.384e-5. The 411-cell comparison across all representations gives mean
region-level changes of -0.10104, -0.13201 and -0.14184. The crossed analysis
uses 22 primary units (paired p=0.42449). On the balanced 105-cell catalogue
panel, the gradient-boosting advantage over the internal determinant-counting
heuristic changes from 0.1248 to 0.0192. This is not a comparison with
ResFinder or AMRFinderPlus.

Cell-level AUROC/AUPRC/error/ECE outputs were inspected directly. They cannot
all be recalculated from individual predictions because fitted models and
per-isolate probabilities are not part of the archive.

## Reproducibility notes

The supplied `requirements.txt` gives lower bounds, not a fully locked
environment. The release-preparation audit was executed with Python 3.12,
NumPy 2.5.4, pandas 3.1.0 and SciPy 1.18.1. Those versions document the audit
environment; they are not asserted as the versions used for the original
model training.
