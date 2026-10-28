# Licence and provenance audit

Date: 2026-10-28.

## Finding

The public CABBAGE article and the EMBL-EBI AMR Portal identify CABBAGE as the
source of the portal data. The article explicitly states that all data,
including genome annotations, are distributed through the EMBL-EBI FTP site
under CC BY 4.0. This is consistent with the authors' declaration that the 78
benchmark files derive exclusively from redistributable CABBAGE material.

No contradiction was found in the file inventory: the release contains
derived arrays, vocabularies, public accessions, metadata and benchmark split
assignments, not copies of the upstream CABBAGE tables or genome assemblies.

## Licence allocation

| Component | Licence | Basis |
|---|---|---|
| Source code and scripts | MIT | Existing author-supplied `LICENSE` |
| Derived 78-file data archive | CC BY 4.0 | CABBAGE licence plus authors' provenance declaration |
| Manuscript and figures | Not assigned by this repository | Governed by authors/OUP publishing terms |

## CC BY 4.0 obligations implemented

- CABBAGE and EMBL-EBI are identified and linked.
- The CC BY 4.0 licence and canonical URL are stated.
- Transformations and benchmark-specific modifications are described.
- No endorsement is implied.
- Code and data licences are separated.

CC BY 4.0 permits sharing and adaptation, including database extraction and
reuse, provided attribution is retained, the licence is linked, and changes
are indicated. It does not license patents, trademarks, privacy rights or
third-party rights that the licensor does not control.

## Decision

Provenance scientifique : **CONFIRMÉE PAR LES AUTEURS**.  
Compatibilité de redistribution : **VALIDÉE après contrôle**.

Residual risk is limited to the accuracy of the authors' declaration that no
separately restricted observations entered the derived archive. That factual
declaration should be retained with the project records.

