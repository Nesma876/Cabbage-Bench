#!/usr/bin/env python3
"""
Fuites par duplication : le meme isolat des deux cotes d'un decoupage.

POURQUOI
  CABBAGE agrege neuf bases publiques — PATRIC/BV-BRC, les antibiogrammes
  NCBI, NARMS, NDARO, le CDC, COMPARE, PathogenWatch, Microreact, PubMLST —
  plus la litterature. Un meme isolat peut donc y figurer plusieurs fois, sous
  des identifiants differents selon la source qui l'a depose.

  Si c'est le cas, AUCUN de nos trois protocoles ne l'attrape. Le decoupage
  par lignee separe des groupes de genomes apparentes, pas des copies du meme
  genome portant deux identifiants ; les decoupages par region et par periode
  encore moins, puisque deux depots du meme isolat peuvent porter des
  metadonnees differentes. Une telle fuite gonflerait TOUTES nos mesures, y
  compris celles qu'on presente comme protegees.

CE QUE LE SCRIPT MESURE
  0. Les colonnes de la table CABBAGE : existe-t-il une colonne de provenance ?
     Sans elle on ne peut pas attribuer une entree a une base amont, et la
     detection doit passer par le contenu.
  1. Un meme assembly_ID sous plusieurs BioSample_ID. C'est la duplication la
     plus directe : le meme genome compte deux fois comme deux isolats.
  2. Des contenus geniques IDENTIQUES entre assemblages distincts, detectes
     par empreinte de l'ensemble de genes. Cout lineaire.
  3. Des contenus QUASI identiques, Jaccard au-dessus d'un seuil, cherches
     SEULEMENT a l'interieur de chaque lignee — deux genomes tres proches y
     sont forcement, donc on ne rate rien et le cout reste modeste.
  4. Et la seule mesure qui compte vraiment : la part des isolats de TEST
     qui ont un jumeau dans l'ENTRAINEMENT, protocole par protocole.

Usage :
    python deduplication.py --bench $BASE/bench
    python deduplication.py --bench $BASE/bench --seuil-jaccard 0.98
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent


def charger_noyau():
    src = (ICI / "cellule_tout.py").read_text(encoding="utf-8")
    src = src.split('print("Pret.")')[0]
    e = {}
    exec(compile(src, "cellule_tout.py", "exec"), e)
    return e


def colonnes_cabbage(N, release="2025-11"):
    """Y a-t-il une colonne de provenance dans la table des phenotypes ?"""
    con = N["_con"]()
    url = f"{N['FTP']}/{release}/phenotype.parquet"
    d = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}') LIMIT 1").df()
    noms = d["column_name"].tolist()
    print(f"\n=== {len(noms)} colonnes dans phenotype.parquet ===")
    for i in range(0, len(noms), 3):
        print("   " + "".join(f"{c:<34}" for c in noms[i:i+3]))
    mots = ("source", "database", "provenance", "origin", "submitter",
            "repository", "study", "project", "publication", "pmid", "doi")
    trouve = [c for c in noms if any(m in c.lower() for m in mots)]
    print(f"\n  colonnes evoquant une provenance : "
          f"{trouve if trouve else 'AUCUNE'}")
    if trouve:
        for c in trouve[:3]:
            v = con.execute(f"SELECT {c}, count(*) n FROM read_parquet('{url}') "
                            f"GROUP BY 1 ORDER BY n DESC LIMIT 8").df()
            print(f"\n  valeurs de {c} :")
            print("    " + v.to_string(index=False).replace("\n", "\n    "))
    return noms, trouve


def charger_cache(chemin):
    with open(chemin, "rb") as f:
        b = pickle.load(f)
    if isinstance(b, dict) and "vocab" in b and "genomes" in b:
        v = b["vocab"]
        return {a: {v[i] for i in arr} for a, arr in b["genomes"].items()}
    return b


def _jaccard(a, b):
    i = len(a & b)
    u = len(a) + len(b) - i
    return i / u if u else 0.0


def doublons(dossier, espece, seuil=0.99):
    d0 = Path(dossier)
    with open(d0 / "jeu.pkl", "rb") as f:
        jeu = pickle.load(f)
    meta = jeu["meta"]
    # Les decoupages se lisent dans splits.csv, PAS dans jeu.pkl : run_all.py
    # ecrit le pickle AVANT d'ajouter la colonne des lignees, si bien que
    # jeu["splits"] n'a que aleatoire, region et periode. Lire le pickle
    # revenait a ne pas mesurer le protocole cense proteger de cette fuite.
    f_sp = d0 / "splits.csv"
    sp = (pd.read_csv(f_sp, index_col=0) if f_sp.exists() else jeu["splits"])
    if "lignee" not in sp.columns:
        print("  ATTENTION : pas de colonne lignee dans splits.csv")
    cache = charger_cache(d0 / "genes.pkl")
    f_lig = d0 / "lignees.csv"
    lig = (pd.read_csv(f_lig, index_col=0)["lignee"] if f_lig.exists() else None)

    print(f"\n=== {espece} : {len(meta)} isolats, {len(cache)} genomes annotes ===")

    # --- 1. un assemblage sous plusieurs identifiants d'echantillon -------
    par_asm = meta.groupby("assembly_ID").size()
    multi = par_asm[par_asm > 1]
    print(f"  assemblages portes par plusieurs isolats : {len(multi)}"
          + (f"  (jusqu'a {int(multi.max())} fois)" if len(multi) else ""))

    # --- 2. contenus geniques strictement identiques ----------------------
    par_emp = defaultdict(list)
    for a, g in cache.items():
        par_emp[hash(frozenset(g))].append(a)
    exact = [v for v in par_emp.values() if len(v) > 1]
    n_exact = sum(len(v) for v in exact)
    print(f"  groupes de contenu genique identique : {len(exact)} "
          f"({n_exact} genomes concernes)")

    # --- 3. quasi-identiques, cherches dans chaque lignee -----------------
    #    Deux genomes de Jaccard >= 0,99 sont forcement dans la meme lignee :
    #    chercher ailleurs serait du calcul perdu.
    # Un seuil unique ne distingue pas un DOUBLON d'un CLONE. Sur une espece
    # tres clonale comme S. enterica, deux genomes d'un meme serovar passent
    # aisement 0,99 de Jaccard sur le contenu accessoire. On balaie donc
    # plusieurs seuils : si les comptes s'effondrent brutalement entre 0,99 et
    # 1,00, il s'agit de duplication ; s'ils decroissent continument, c'est de
    # la structure clonale, et c'est le decoupage par lignee qui en repond.
    paires, spectre = [], {}
    if lig is not None:
        seuils = sorted({0.95, 0.98, 0.99, 0.995, 1.0, seuil})
        cpt = {s: 0 for s in seuils}
        for _, grp in lig.groupby(lig):
            accs = [a for a in grp.index if a in cache]
            if len(accs) < 2 or len(accs) > 1500:
                continue
            for i in range(len(accs)):
                gi = cache[accs[i]]
                for j in range(i + 1, len(accs)):
                    v = _jaccard(gi, cache[accs[j]])
                    for s in seuils:
                        if v >= s: cpt[s] += 1
                    if v >= seuil:
                        paires.append((accs[i], accs[j]))
        spectre = cpt
        print("  paires par seuil de Jaccard : "
              + "  ".join(f"{s:g} -> {cpt[s]}" for s in seuils))
    print(f"  paires retenues comme jumeaux (>= {seuil}) : {len(paires)}")

    # --- groupes de jumeaux, par union des relations ----------------------
    parent = {}
    def trouver(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def unir(x, y):
        rx, ry = trouver(x), trouver(y)
        if rx != ry: parent[rx] = ry
    for v in exact:
        for b in v[1:]: unir(v[0], b)
    for a, b in paires: unir(a, b)
    groupes = defaultdict(list)
    for a in parent: groupes[trouver(a)].append(a)
    groupes = {k: v for k, v in groupes.items() if len(v) > 1}
    print(f"  => {len(groupes)} groupes de jumeaux, "
          f"{sum(len(v) for v in groupes.values())} genomes")

    # --- 4. ces jumeaux traversent-ils les decoupages ? -------------------
    asm = meta["assembly_ID"]
    lignes = []
    for dec in [c for c in ("aleatoire", "lignee", "region", "periode")
                if c in sp.columns]:
        col = sp[dec].astype(str)
        tr_a = set(asm[(col == "train").values].dropna())
        te_i = (col == "test").values
        n_te = int(te_i.sum())
        if not n_te:
            continue
        touches = 0
        for a in asm[te_i].dropna():
            r = parent.get(a)
            if r is None:
                continue
            if any(x in tr_a for x in groupes.get(r, ()) if x != a):
                touches += 1
        lignes.append({"espece": espece, "decoupage": dec, "n_test": n_te,
                       "test_avec_jumeau_train": touches,
                       "pct": round(100 * touches / n_te, 2)})
        print(f"  {dec:10s} {touches:5d} / {n_te:5d} isolats de test ont un "
              f"jumeau en entrainement  ({100*touches/n_te:5.2f} %)")
    return pd.DataFrame(lignes)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    p.add_argument("--seuil-jaccard", type=float, default=0.99)
    p.add_argument("--sans-colonnes", action="store_true",
                   help="ne pas interroger la table distante")
    a = p.parse_args()

    N = charger_noyau()
    if not a.sans_colonnes:
        try:
            colonnes_cabbage(N)
        except Exception as e:                                   # noqa: BLE001
            print(f"\nLecture des colonnes impossible : {type(e).__name__}: {e}")

    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "jeu.pkl").exists()))
    tous = []
    for d in dossiers:
        try:
            t = doublons(d, d.name, a.seuil_jaccard)
            if not t.empty: tous.append(t)
        except Exception as e:                                   # noqa: BLE001
            print(f"{d.name} : ECHEC — {type(e).__name__}: {e}")

    if tous:
        g = pd.concat(tous, ignore_index=True)
        g.to_csv(racine / "doublons_tous.csv", index=False)
        print("\n=== Part des isolats de test ayant un jumeau en entrainement ===")
        print(g.pivot_table(index="espece", columns="decoupage",
                            values="pct").round(2).to_string())
        pire = g["pct"].max()
        print(f"\n  maximum observe : {pire:.2f} %")
        print("""
  COMMENT LIRE CE TABLEAU, et ce qu'il ne dit pas.

  Un taux eleve sous 'aleatoire' n'est pas un defaut du benchmark : c'est
  precisement la fuite clonale que l'article denonce, enfin chiffree. Ce qui
  compte est la COLONNE 'lignee' : si elle est nettement plus basse que les
  autres, le protocole fait son travail.

  Et le seuil de Jaccard ne separe pas un doublon d'un clone. Comparer la
  ligne 'contenu identique' aux paires a 0,99 le montre : quand les secondes
  sont cent fois plus nombreuses, on mesure de la clonalite, pas de la
  duplication. Seule la premiere justifierait une deduplication.""")


if __name__ == "__main__":
    main()
