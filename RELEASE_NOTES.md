# CABBAGE-Bench v1.0.0

Official first scientific release. Repository:
<https://github.com/Nesma876/Cabbage-Bench/tree/v1.0.0>. Dataset DOI:
<https://doi.org/10.5281/zenodo.23732198>.

This first scientific release contains the benchmark construction
and evaluation code, five analysis scripts, HPC launch material, archived
result tables, documentation, a citation file, and an executable release
validator. The companion Zenodo record is intended to contain exactly
`cabbage-bench-depot.zip` and `cabbage-bench-donnees.tgz`.

Corrections made during release preparation:

- removed machine-specific absolute paths from analysis scripts;
- made the AMR baseline load its vocabulary from the released
  `genes_pan.csv` instead of the unpublished `jeu.pkl` object;
- corrected the crossed decomposition test to average repeated model-family
  measurements before testing the 22 primary units (paired p = 0.4245);
- added integrity, matrix, mask, partition, and headline-result checks.

No scientific value was changed to force agreement with the manuscript.

The final archive SHA-256 is recorded outside the archive in
`DEPOSIT_REPORT.md` to avoid a self-referential checksum. The original
received archive remains preserved separately with SHA-256
`2d56339a252ce76fe7324cd786a63f6e1f3baf13260b2799683bde4668f411f0`.

