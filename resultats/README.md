# Result files

Every number in the paper is computed from these files by the scripts in
`analyse/`. Representation prefixes: `amr` = resistance catalogue,
`pan` = annotated gene content, `pg` = orthologue families.

| file | contents |
|---|---|
| `*_resultats_forts_tous.csv` | AUROC, AUPRC, major and very major error rates, ECE, per (species, protocol, model, drug) |
| `*_loro_tous.csv` | leave-one-region-out: `type` is `region` (held out) or `temoin` (size-matched control), `tirage` indexes the control draws |
| `amr_loro_regions_tous.csv` | per held-out region: isolate count, drug count, share of lineages already seen in training |
| `decomposition_tous.csv` | crossed holdout. `analyse` is `geographie` (provenance fixed, a region held out) or `provenance` (region fixed, a provenance level held out); `strate` is the fixed level, `tenu` the held-out one |
| `desaccord_normes.csv` | CLSI vs EUCAST phenotype disagreement per species and antimicrobial |
| `provenance_*.csv` | coverage, confounding with region (Cramer's V) and decomposability of each provenance column |
| `pangenome.csv` | orthologue family counts from the de novo clustering |

Column names are in French, as in the code: `espece` species,
`antibiotique` antimicrobial, `decoupage` protocol (`aleatoire` random,
`lignee` lineage-aware, `region` geography-aware, `periode` time-aware),
`modele` model (`regles` rule-based, `logistique` logistic regression,
`rf` random forest, `gb` gradient boosting), `chute`/`delta` the change
against a control, `err_majeure_pct` major errors, `err_tres_majeure_pct`
very major errors, `tenu` held out, `temoin` control.
