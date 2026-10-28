#!/usr/bin/env python3
"""
Refait le regroupement en lignees avec un critere de calibration corrige.

LE BUG
  La calibration exigeait DEUX conditions simultanees :
      la plus grosse lignee <= 15 % des genomes       (ce qui protege)
      >= 95 % des genomes dans un groupe de taille >= 5  (la couverture)
  et, si aucun seuil ne tenait les deux, se rabattait sur 0,45 en aveugle.

  Chez A. baumannii et P. aeruginosa, dont les genomes sont beaucoup plus
  homogenes en contenu accessoire (distance de Jaccard mediane 0,311 et 0,239
  contre 0,406 chez K. pneumoniae), aucun seuil ne tenait les deux. Le repli a
  donc produit une lignee contenant 55 % des genomes. Un decoupage par lignee
  qui met 55 % des genomes dans un seul groupe ne protege de rien : soit ce
  bloc est dans l'entrainement et le test n'est qu'un residu, soit l'inverse.

LA CORRECTION
  Les deux criteres ne sont pas de meme nature. La taille de la plus grosse
  lignee est IMPERATIVE : c'est elle qui fait la validite du decoupage. La
  couverture est SOUHAITABLE : des singletons partent simplement en
  entrainement, ce qui ne fausse rien, cela reduit seulement la taille utile
  du test.

  On parcourt donc les seuils, on garde ceux qui respectent le critere
  imperatif, et parmi eux on choisit celui qui maximise la couverture. Si
  aucun ne respecte l'imperatif, on prend celui qui minimise la plus grosse
  lignee et on le dit clairement.

Usage :
    python refaire_lignees.py --bench $BASE/bench \
        --especes acinetobacter_baumannii pseudomonas_aeruginosa

Ce script REECRIT lignees.csv, splits.csv et resultats.csv des especes
traitees. Les anciennes versions sont sauvees en .avant.
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy import sparse

ICI = Path(__file__).resolve().parent


def charger_noyau():
    src = (ICI / "cellule_tout.py").read_text(encoding="utf-8")
    src = src.split('print("Pret.")')[0]
    espace = {}
    exec(compile(src, "cellule_tout.py", "exec"), espace)
    return espace


def charger_cache(chemin):
    """Accepte le format compact (vocab + indices) et l'ancien (ensembles)."""
    with open(chemin, "rb") as f:
        brut = pickle.load(f)
    if isinstance(brut, dict) and "vocab" in brut and "genomes" in brut:
        v = brut["vocab"]
        return {a: {v[i] for i in arr} for a, arr in brut["genomes"].items()}
    return brut


def lignees_corrigees(cache, genes_amr=(), bande=(0.01, 0.99),
                      max_part=15, couverture_min=60, graine=0, verbeux=True):
    accs = sorted(cache)
    amr = {g.lower() for g in genes_amr}
    tous = sorted({g for s in cache.values() for g in s if g.lower() not in amr})
    idx = {g: i for i, g in enumerate(tous)}
    lignes, cols = [], []
    for i, a in enumerate(accs):
        for g in cache[a]:
            j = idx.get(g)
            if j is not None:
                lignes.append(i); cols.append(j)
    X = sparse.csr_matrix((np.ones(len(lignes), np.int8), (lignes, cols)),
                          shape=(len(accs), len(tous)))
    part = np.asarray(X.mean(axis=0)).ravel()
    garde = (part >= bande[0]) & (part <= bande[1])
    X = X[:, garde]
    n = X.shape[0]
    if verbeux:
        print(f"  {n} assemblages, {X.shape[1]} genes accessoires "
              f"({len(amr)} genes de resistance exclus)")

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X.toarray().astype(bool), metric="jaccard", n_jobs=-1)
    med = float(np.median(D[np.triu_indices(n, 1)]))
    Z = linkage(squareform(D, checks=False), method="complete")
    del D
    if verbeux:
        print(f"  distance de Jaccard mediane : {med:.3f}")

    # --- la correction : l'imperatif d'abord, la couverture ensuite --------
    essais = []
    for s in np.arange(0.10, 0.72, 0.02):
        lab = fcluster(Z, t=s, criterion="distance")
        t = pd.Series(lab).value_counts()
        essais.append({
            "seuil": round(float(s), 3),
            "groupes": int(t.size),
            "pct_max": round(100 * t.iloc[0] / n, 1),
            "pct_utiles": round(100 * t[t >= 5].sum() / n, 1),
            "mediane": int(t.median()),
            "lab": lab,
        })
    tab = pd.DataFrame([{k: v for k, v in e.items() if k != "lab"}
                        for e in essais])

    ok = [e for e in essais if e["pct_max"] <= max_part]
    if ok:
        choix = max(ok, key=lambda e: e["pct_utiles"])
        print(f"  seuil retenu {choix['seuil']:.2f} : plus grosse lignee "
              f"{choix['pct_max']} % (<= {max_part} %), couverture "
              f"{choix['pct_utiles']} %")
    else:
        choix = min(essais, key=lambda e: e["pct_max"])
        print(f"  AUCUN seuil ne tient {max_part} % ; le meilleur possible est "
              f"{choix['pct_max']} % au seuil {choix['seuil']:.2f}.")

    # --- verdict -----------------------------------------------------------
    # Les deux facons de rater sont symetriques, et il faut nommer les deux :
    #   une lignee geante -> le decoupage ne protege rien
    #   une couverture basse -> presque tout devient singleton, et tenir des
    #   singletons hors de l'entrainement equivaut a un tirage aleatoire,
    #   puisque leurs plus proches voisins y restent.
    # Quand l'un des deux arrive, le chiffre existe mais il ne veut rien dire.
    choix["fiable"] = (choix["pct_max"] <= max_part
                       and choix["pct_utiles"] >= couverture_min)
    if not choix["fiable"]:
        raison = ("une lignee concentre "
                  f"{choix['pct_max']} % des genomes"
                  if choix["pct_max"] > max_part else
                  f"seulement {choix['pct_utiles']} % des genomes sont dans un "
                  f"groupe d'au moins 5, le reste etant des singletons")
        print(f"  VERDICT : decoupage par lignee NON FIABLE pour cette espece "
              f"— {raison}.")
        print("  A declarer comme tel dans l'article. Le contenu genique ne "
              "suffit pas a definir des lignees ici ; il faudrait un typage "
              "MLST ou un arbre phylogenetique sur genes de menage.")
    else:
        print("  VERDICT : decoupage par lignee exploitable.")

    print(tab.to_string(index=False))
    s = pd.Series(choix["lab"], index=accs, name="lignee")
    t = s.value_counts()
    print(f"  => {t.size} lignees ; la plus grande = {t.iloc[0]} genomes "
          f"({100*t.iloc[0]/n:.1f} %), mediane {int(t.median())}, "
          f"{100*t[t>=5].sum()/n:.1f} % dans un groupe d'au moins 5")
    return s, choix, tab


def traiter(dossier, N, max_part, couverture_min, refaire_resultats=True):
    d0 = Path(dossier)
    print(f"\n=== {d0.name} ===")
    with open(d0 / "jeu.pkl", "rb") as f:
        d = pickle.load(f)
    cache = charger_cache(d0 / "genes.pkl")

    prefixes = ("bla", "aac", "aph", "ant", "sul", "tet", "erm", "mcr",
                "qnr", "van", "oqx", "dfr", "cat", "mph", "arm", "rmt")
    tous_genes = {g for v in cache.values() for g in v}
    amr = set(d["genes"]) | {g for g in tous_genes
                             if g.lower().startswith(prefixes)}

    lignees, choix, tab = lignees_corrigees(cache, genes_amr=amr,
                                            max_part=max_part,
                                            couverture_min=couverture_min)
    for f in ("lignees.csv", "splits.csv", "resultats.csv"):
        if (d0 / f).exists():
            shutil.copy2(d0 / f, d0 / (f + ".avant"))
    lignees.to_csv(d0 / "lignees.csv")
    tab.to_csv(d0 / "calibration_lignees.csv", index=False)

    d["splits"]["lignee"] = N["splits_par_lignee"](d["meta"], lignees)
    fuite = N["controle_fuite"](d["meta"], lignees, d["splits"])
    d["splits"].to_csv(d0 / "splits.csv")
    with open(d0 / "jeu.pkl", "wb") as f:
        pickle.dump(d, f, protocol=4)

    if refaire_resultats:
        res = N["evaluer"](d)
        res.insert(0, "espece", d.get("espece", d0.name))
        res.to_csv(d0 / "resultats.csv", index=False)

    return {"espece": d0.name, "lignees": int(lignees.nunique()),
            "seuil": choix["seuil"], "pct_max": choix["pct_max"],
            "couverture_pct": choix["pct_utiles"],
            "fiable": choix["fiable"], "fuite": len(fuite)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="+", required=True)
    p.add_argument("--max-part", type=float, default=15,
                   help="part maximale de la plus grosse lignee, en pourcent")
    p.add_argument("--couverture-min", type=float, default=60,
                   help="couverture minimale sous laquelle le decoupage par "
                        "lignee est declare non fiable")
    p.add_argument("--sans-resultats", action="store_true",
                   help="refaire seulement les lignees et les decoupages")
    a = p.parse_args()

    N = charger_noyau()
    bilan = []
    for e in a.especes:
        d = Path(a.bench) / e
        if not (d / "genes.pkl").exists():
            print(f"{e} : genes.pkl absent, on saute")
            continue
        bilan.append(traiter(d, N, a.max_part, a.couverture_min,
                             not a.sans_resultats))

    if bilan:
        print("\n=== Bilan ===")
        print(pd.DataFrame(bilan).to_string(index=False))


if __name__ == "__main__":
    main()
