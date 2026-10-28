#!/usr/bin/env python3
"""Reconstruit X : contenu genique du genome entier + bloc catalogue AMR.

LE DEFAUT QUE CE SCRIPT CORRIGE
  La matrice initiale du benchmark etait construite a partir de la seule
  colonne amr_element_symbol de CABBAGE (cellule_tout.py:211). Autrement dit,
  tous les modeles ne voyaient QUE des genes de resistance. Deux consequences
  que l'article annoncait a tort :

  1. La comparaison « apprentissage contre regles de catalogue » n'en etait
     pas une. Les deux bras lisaient le meme catalogue, l'un avec des poids
     appris, l'autre par comptage. L'avance de 0,190 en AUROC et son
     effondrement sous decalage ne pouvaient donc rien dire sur l'interet
     d'apprendre plutot que de consulter une nomenclature.
  2. Le mecanisme invoque pour la fuite clonale — « des genomes apparentes
     partagent un contenu accessoire qui sert d'identifiant » — ne pouvait pas
     operer, puisque le contenu accessoire n'etait pas dans la matrice.

CE QUE LA NOUVELLE MATRICE CONTIENT
  Deux blocs, repris dans genes_pan.csv par une colonne `bloc` :

  - `pan` : presence/absence des genes nommes des GFF, c'est-a-dire le contenu
    genique a l'echelle du genome. Les suffixes de copie sont fusionnes
    (bla_1, bla_2, bla_3 -> bla) : ce sont des indices de paralogues poses par
    le pipeline d'annotation, pas des genes differents, et les garder separes
    rendait chacun rare alors que le gene est courant. On retient la bande de
    prevalence 1-99 % et un minimum de 10 genomes, pour ecarter a la fois le
    coeur invariant (sans pouvoir discriminant) et la queue de singletons
    (bruit d'annotation).
  - `amr` : le catalogue, inchange, tel que l'ancienne matrice le fournissait.

  Les modeles appris lisent les deux blocs. Le baseline a regles ne lit que
  `amr`, parce que les noms des GFF sont generiques : `bla_1` ne dit pas de
  quelle beta-lactamase il s'agit, et une regle clinique a besoin de le
  savoir. C'est ce que fait le masque pose par baselines_forts.charger.

CE QU'ELLE NE CORRIGE PAS, ET QU'IL FAUDRA DIRE
  _genes_gff ne retient que les champs Name= et gene= des GFF. Tous les CDS
  annotes « hypothetical protein » disparaissent, et ils sont majoritaires.
  On obtient donc de l'ordre de 800 genes accessoires nommes par espece, la
  ou un vrai pangenome de K. pneumoniae en compte 10 000 a 20 000. C'est un
  proxy du genome accessoire, meilleur que le catalogue seul mais pas
  equivalent a un regroupement en familles d'orthologues. Le niveau 2 du plan
  (MMseqs2 sur les proteines du bloc ##FASTA) leve cette limite.

Usage :
    python refaire_matrice.py --bench $BASE/bench
    python refaire_matrice.py --bench $BASE/bench --min-prev 0.01 --max-prev 0.99
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SUFFIXE = re.compile(r"_\d+$")


def charger_contenu(chemin):
    """assembly_ID -> ensemble de noms de genes, suffixes de copie fusionnes."""
    with open(chemin, "rb") as f:
        b = pickle.load(f)
    if isinstance(b, dict) and "vocab" in b and "genomes" in b:
        v = b["vocab"]
        brut = {a: {v[i] for i in arr} for a, arr in b["genomes"].items()}
    else:
        brut = b
    return {a: {SUFFIXE.sub("", g) for g in gs} for a, gs in brut.items()}


def traiter(dossier, espece, min_prev, max_prev, min_n):
    d0 = Path(dossier)
    with open(d0 / "jeu.pkl", "rb") as f:
        jeu = pickle.load(f)
    meta = jeu["meta"]
    X_amr = jeu["X"]
    X_amr = X_amr.tocsr() if sparse.issparse(X_amr) else sparse.csr_matrix(X_amr)
    genes_amr = list(jeu["genes"])
    contenu = charger_contenu(d0 / "genes.pkl")

    # L'ordre des lignes doit rester EXACTEMENT celui de jeu.pkl, sinon Y, M
    # et splits.csv ne correspondent plus. On passe par assembly_ID.
    asm = meta["assembly_ID"]
    n_iso = len(meta)
    sans = int(asm.isna().sum() + sum(1 for a in asm.dropna() if a not in contenu))

    voc = Counter()
    for a in asm.dropna():
        g = contenu.get(a)
        if g:
            voc.update(g)
    n_annotes = sum(1 for a in asm.dropna() if a in contenu)
    lo, hi = min_prev * n_annotes, max_prev * n_annotes
    garde = sorted(k for k, v in voc.items() if lo <= v <= hi and v >= min_n)
    if not garde:
        print(f"{espece} : aucun gene retenu, ignore")
        return None
    ii = {g: j for j, g in enumerate(garde)}

    li, co = [], []
    for i, a in enumerate(asm):
        if a is None or a != a:
            continue
        for g in contenu.get(a, ()):
            j = ii.get(g)
            if j is not None:
                li.append(i); co.append(j)
    X_pan = sparse.csr_matrix(
        (np.ones(len(li), np.int8), (li, co)), shape=(n_iso, len(garde)))

    X = sparse.hstack([X_pan, X_amr], format="csr")
    voc_out = pd.DataFrame(
        {"gene": garde + genes_amr,
         "bloc": ["pan"] * len(garde) + ["amr"] * len(genes_amr)})
    assert X.shape[1] == len(voc_out), (X.shape, len(voc_out))

    sparse.save_npz(d0 / "X_genes_pan.npz", X)
    voc_out.to_csv(d0 / "genes_pan.csv", index=False)

    dens = X_pan.getnnz(axis=1)
    print(f"\n=== {espece} ===")
    print(f"  isolats {n_iso}   annotes {n_annotes}"
          + (f"   sans contenu genique {sans}" if sans else ""))
    print(f"  vocabulaire GFF apres fusion des suffixes : {len(voc)}")
    print(f"  retenus (prevalence {min_prev:g}-{max_prev:g}, "
          f">= {min_n} genomes) : {len(garde)}")
    print(f"  catalogue AMR : {len(genes_amr)}")
    print(f"  X final : {X.shape[0]} x {X.shape[1]}   "
          f"densite {100 * X.getnnz() / (X.shape[0] * X.shape[1]):.2f} %")
    print(f"  genes pan actifs par isolat : mediane {int(np.median(dens))}  "
          f"min {int(dens.min())}  max {int(dens.max())}")
    return {"espece": espece, "isolats": n_iso, "annotes": n_annotes,
            "voc_gff": len(voc), "pan": len(garde), "amr": len(genes_amr),
            "total": X.shape[1],
            "densite_pct": round(100 * X.getnnz() / (X.shape[0] * X.shape[1]), 2),
            "pan_median_par_isolat": int(np.median(dens))}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    p.add_argument("--min-prev", type=float, default=0.01)
    p.add_argument("--max-prev", type=float, default=0.99)
    p.add_argument("--min-genomes", type=int, default=10)
    p.add_argument("--garder-suffixes", action="store_true",
                   help="ne pas fusionner bla_1, bla_2 ... (pour comparaison)")
    a = p.parse_args()

    if a.garder_suffixes:
        global SUFFIXE
        SUFFIXE = re.compile(r"$^")        # ne correspond a rien

    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "jeu.pkl").exists()))
    lignes = []
    for d in dossiers:
        try:
            r = traiter(d, d.name, a.min_prev, a.max_prev, a.min_genomes)
            if r:
                lignes.append(r)
        except Exception as e:                                   # noqa: BLE001
            print(f"{d.name} : ECHEC — {type(e).__name__}: {e}")

    if lignes:
        t = pd.DataFrame(lignes)
        t.to_csv(racine / "matrices_pan.csv", index=False)
        print("\n=== Nouvelles matrices ===")
        print(t.to_string(index=False))
        print(f"""
  Relancer ensuite avec CABBAGE_MATRICE=pan :

    CABBAGE_MATRICE=pan python baselines_forts.py --bench {racine}
    CABBAGE_MATRICE=pan python geographie.py --bench {racine}

  Sans cette variable, les scripts reprennent l'ancienne matrice catalogue,
  ce qui permet de comparer les deux et de chiffrer ce que le contenu
  genomique apporte reellement.""")


if __name__ == "__main__":
    main()
