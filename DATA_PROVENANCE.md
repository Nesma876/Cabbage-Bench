# Data provenance

## Source

CABBAGE-Bench is derived from the Comprehensive Assessment of Bacterial-Based
Antimicrobial resistance prediction from GEnotypes (CABBAGE) database,
distributed through the EMBL-EBI Antimicrobial Resistance Portal. The frozen
upstream version used by this benchmark is release `2025-11`.

- Portal: https://www.ebi.ac.uk/amr/
- Release directory: https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases/
- Source article: Dickens E. et al. *A comprehensive AMR
  genotype--phenotype database (CABBAGE).* Nucleic Acids Research (2026),
  https://doi.org/10.1093/nar/gkag780
- Upstream licence: CC BY 4.0.

The CABBAGE article states that all portal data, including genome annotations,
are available from the EMBL-EBI FTP site under CC BY 4.0. The authors of this
benchmark additionally confirm that every one of the 78 released files comes
exclusively from redistributable CABBAGE material and contains no observation
subject to an additional restriction.

## Transformations

The release does not redistribute the upstream Parquet tables, assemblies or
GFF files. It provides derived benchmark representations:

- binary multi-label resistance matrices and observed-value masks;
- antimicrobial column names and public BioSample accessions;
- selected isolate metadata used to define region, period and provenance;
- accessory-content lineage assignments;
- three sparse genomic feature representations with vocabularies;
- fixed train/validation/test assignments for random, lineage-aware,
  geography-aware and time-aware protocols.

These files result from selection of six species, phenotype filtering,
standardisation, explicit preservation of missing labels, feature extraction,
lineage construction and benchmark partitioning. They are therefore marked as
modified/derived material, as required by CC BY 4.0.

## Responsibility statement

Provenance scientifique : **CONFIRMÉE PAR LES AUTEURS**.  
Compatibilité de redistribution : **VALIDÉE après contrôle documentaire**.

This is a provenance and licence-compliance record, not independent legal
advice. Downstream users remain responsible for complying with CC BY 4.0 and
for retrieving upstream assemblies or annotations under the terms displayed
by their respective repositories.

