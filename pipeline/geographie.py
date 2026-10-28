#!/usr/bin/env python3
"""
Generalisation geographique : leave-one-region-out, et ses facteurs de confusion.

POURQUOI CE SCRIPT
  Le decoupage geographique du benchmark tient UNE seule region hors de
  l'entrainement, la deuxieme la mieux fournie. C'est un tirage unique. Or la
  geographie s'est revelee le decalage dominant dans quatre especes sur six :
  l'effet principal de l'article repose donc sur un seul tirage, ce qu'un
  relecteur ne laissera pas passer. On remplace par un leave-one-region-out
  sur toutes les regions assez peuplees, et on rapporte la DISTRIBUTION des
  chutes plutot qu'un chiffre.

  Le script repond en meme temps aux deux objections previsibles.

  CONFUSION 1, la prevalence. Si la region tenue a l'ecart a simplement plus
  ou moins de resistance, la chute d'AUROC peut n'etre qu'un effet de
  deseequilibre de classes, qu'une recalibration absorberait. On correle donc
  la chute avec le deplacement de prevalence, antibiotique par antibiotique.

  CONFUSION 2, la lignee. Si la region de test contient des lignees absentes
  de l'entrainement, l'effet « geographie » est un effet lignee deguise. On
  mesure donc, pour chaque region, la part de ses isolats dont la lignee est
  AUSSI presente dans l'entrainement. Une part elevee et une chute forte : la
  geographie agit par elle-meme. Une part faible : les deux effets sont
  confondus et il faut le dire.

Usage :
    python geographie.py --bench $BASE/bench
    python geographie.py --bench $BASE/bench --especes klebsiella_pneumoniae \
        --min-isolats 150
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression

from baselines_forts import _mesures, charger, colonnes_train


def regions_utiles(meta, min_isolats):
    r = meta["region"].astype("object").where(meta["region"].notna(), None)
    c = r.dropna().value_counts()
    return r, c[c >= min_isolats]


def part_lignees_vues(meta, lignees, tr, te):
    """Part des isolats de test dont la lignee apparait aussi en entrainement."""
    if lignees is None:
        return np.nan
    lig = meta["assembly_ID"].map(lignees)
    vues = set(lig[tr].dropna().unique())
    lt = lig[te].dropna()
    if not len(lt):
        return np.nan
    return round(100 * lt.isin(vues).mean(), 1)


def un_modele(nom, Xtr, ytr, Xte, yte, rare):
    if nom == "logistique":
        m = LogisticRegression(max_iter=3000, C=1.0)
    elif nom == "rf":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   n_jobs=-1, random_state=0)
    else:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=rare >= 20, validation_fraction=0.15,
            n_iter_no_change=20, random_state=0)
    m.fit(Xtr, ytr)
    return _mesures(m.predict_proba(Xte)[:, 1], yte)


def traiter(dossier, espece, min_isolats=150, n_min_test=20,
            modeles=("logistique", "gb"), n_temoins=5):
    X, Y, M, splits, atb, genes = charger(dossier)
    with open(Path(dossier) / "jeu.pkl", "rb") as f:
        d = pickle.load(f)
    meta = d["meta"]
    Xd = X.toarray().astype(np.float32)

    f_lig = Path(dossier) / "lignees.csv"
    lignees = (pd.read_csv(f_lig, index_col=0)["lignee"]
               if f_lig.exists() else None)

    reg, cnt = regions_utiles(meta, min_isolats)
    print(f"\n=== {espece} : {len(meta)} isolats, "
          f"{len(cnt)} regions de {min_isolats} isolats ou plus ===")
    if cnt.empty:
        print("  aucune region exploitable")
        return pd.DataFrame(), pd.DataFrame()
    print("  " + ", ".join(f"{k} ({v})" for k, v in cnt.items()))

    # Reference : entrainement et test tires au hasard, a taille de test egale
    # a la moyenne des regions, pour que la comparaison soit honnete.
    rng = np.random.default_rng(0)
    lignes, bilan = [], []

    for region, n_reg in cnt.items():
        te = (reg == region).fillna(False).to_numpy()
        tr = ~te & reg.notna().to_numpy()      # regions connues seulement
        if te.sum() < 50 or tr.sum() < 200:
            continue
        # Plusieurs temoins de MEME taille au lieu d'un seul. Avec un tirage
        # unique, le bruit d'echantillonnage de ce tirage entrait dans chacune
        # des differences de la region, et les differences d'une meme region
        # etaient donc correlees : un Wilcoxon sur l'ensemble des paires
        # supposait a tort leur independance. On moyenne l'AUROC du temoin sur
        # plusieurs tirages ; l'inference de reference reste neanmoins celle
        # au niveau de la region.
        temoins = []
        idx_dispo = np.where(reg.notna())[0]
        for _ in range(n_temoins):
            te_i = np.zeros(len(meta), bool)
            te_i[rng.choice(idx_dispo, int(te.sum()), replace=False)] = True
            temoins.append((~te_i & reg.notna().to_numpy(), te_i))

        part_vues = part_lignees_vues(meta, lignees, tr, te)
        # Vocabulaire choisi sur l'entrainement de CE decoupage, pas sur tous
        # les genomes : sinon le choix des caracteristiques connait le contenu
        # de la region tenue a l'ecart.
        Xr = Xd[:, colonnes_train(X, tr)]
        Xt = [Xd[:, colonnes_train(X, t_i)] for t_i, _ in temoins]
        n_atb = 0
        for j, nom in enumerate(atb):
            paires = [("region", tr, te, Xr)]
            for k, (tr_i, te_i) in enumerate(temoins):
                paires.append((f"temoin{k}", tr_i, te_i, Xt[k]))
            for etiquette, mtr, mte, Xu in paires:
                a = mtr & (M[:, j] > 0)
                b = mte & (M[:, j] > 0)
                if b.sum() < n_min_test or len(np.unique(Y[b, j])) < 2:
                    continue
                if a.sum() < 30 or len(np.unique(Y[a, j])) < 2:
                    continue
                rare = int(min((Y[a, j] == 0).sum(), (Y[a, j] == 1).sum()))
                if rare < 5:
                    continue
                for mod in modeles:
                    try:
                        m = un_modele(mod, Xu[a], Y[a, j], Xu[b], Y[b, j], rare)
                    except Exception:                            # noqa: BLE001
                        continue
                    lignes.append({
                        "espece": espece, "region": region,
                        "type": "region" if etiquette == "region" else "temoin",
                        "tirage": 0 if etiquette == "region"
                                  else int(etiquette[6:]),
                        "modele": mod, "antibiotique": nom,
                        "n_test": int(b.sum()),
                        "prev_train_pct": round(100 * Y[a, j].mean(), 1),
                        "prev_test_pct": round(100 * Y[b, j].mean(), 1), **m})
                    if etiquette == "region" and mod == modeles[0]:
                        n_atb += 1
        bilan.append({"espece": espece, "region": region,
                      "isolats": int(n_reg), "antibiotiques": n_atb,
                      "lignees_vues_pct": part_vues})
        print(f"  {str(region)[:28]:28s} {int(n_reg):6d} isolats, "
              f"{n_atb:3d} antibiotiques, "
              f"lignees deja vues {part_vues} %")

    res = pd.DataFrame(lignes)
    bil = pd.DataFrame(bilan)
    if not res.empty:
        res.to_csv(Path(dossier) / "loro.csv", index=False)
        bil.to_csv(Path(dossier) / "loro_regions.csv", index=False)
    return res, bil


def resumer(res, bil):
    """La distribution des chutes, et les deux tests de confusion."""
    if res.empty:
        return
    # appariement region <-> temoin sur (region, modele, antibiotique)
    cle = ["espece", "region", "modele", "antibiotique"]
    # aggfunc="mean" moyenne les plusieurs tirages du temoin : une seule
    # valeur de reference par cellule, debarrassee du bruit d'un tirage unique.
    g = res.pivot_table(index=cle, columns="type", values="auroc",
                        aggfunc="mean")
    # Le pivot cree des colonnes nommees "region" et "temoin", or "region" est
    # deja un niveau d'index : reset_index() refuserait de les faire coexister.
    g = g.rename(columns={"region": "auroc_region", "temoin": "auroc_temoin"})
    g = g.dropna(subset=[c for c in ("auroc_region", "auroc_temoin")
                         if c in g.columns])
    if g.empty or "auroc_temoin" not in g.columns:
        print("\nAucune paire region/temoin comparable.")
        return
    g["chute"] = g["auroc_region"] - g["auroc_temoin"]

    print("\n=== Chute d'AUROC par region, contre un temoin aleatoire "
          "de meme taille ===")
    par_region = (g.reset_index()
                   .groupby(["espece", "region", "modele"])["chute"]
                   .agg(["count", "median", "mean", "min", "max"]).round(3))
    print(par_region.to_string())

    print("\n=== Distribution sur l'ensemble des regions ===")
    print(g.reset_index().groupby(["espece", "modele"])["chute"]
           .agg(["count", "median", "mean", "std",
                 lambda s: (s < 0).mean().round(2)])
           .rename(columns={"<lambda_0>": "part_degradee"}).round(3).to_string())

    # --- confusion 1 : la prevalence ------------------------------------
    p = res[res["type"] == "region"].set_index(cle)
    j = g.join(p[["prev_train_pct", "prev_test_pct", "n_test"]], how="inner")
    j["d_prev"] = (j["prev_test_pct"] - j["prev_train_pct"]).abs()
    print("\n=== Confusion 1 : la chute suit-elle le deplacement de prevalence ? ===")
    for (esp, mod), s in j.reset_index().groupby(["espece", "modele"]):
        if len(s) > 10:
            r = s["chute"].corr(s["d_prev"])
            print(f"  {esp:26s} {mod:11s} r = {r:+.3f}  (n = {len(s)})")
    print("  Un r proche de zero signifie que la chute n'est PAS un effet de")
    print("  prevalence : c'est la relation genotype-phenotype qui ne tient pas.")

    # --- confusion 2 : la lignee ----------------------------------------
    if not bil.empty and bil["lignees_vues_pct"].notna().any():
        m = (g.reset_index().groupby(["espece", "region"])["chute"].median()
              .reset_index()
              .merge(bil, on=["espece", "region"], how="inner"))
        print("\n=== Confusion 2 : la chute suit-elle la nouveaute des lignees ? ===")
        print(m[["espece", "region", "isolats", "lignees_vues_pct", "chute"]]
              .round(3).to_string(index=False))
        if len(m) > 4 and m["lignees_vues_pct"].notna().sum() > 4:
            r = m["chute"].corr(m["lignees_vues_pct"])
            print(f"  correlation chute / lignees deja vues : r = {r:+.3f}")
            print("  Un r positif fort voudrait dire que les regions qui chutent")
            print("  le plus sont celles aux lignees inconnues : l'effet")
            print("  geographique serait alors un effet lignee deguise.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    p.add_argument("--min-isolats", type=int, default=150,
                   help="taille minimale d'une region pour etre tenue a l'ecart")
    p.add_argument("--modeles", nargs="+", default=["logistique", "gb"])
    p.add_argument("--temoins", type=int, default=5,
                   help="nombre de tirages du temoin apparie en taille")
    a = p.parse_args()

    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "jeu.pkl").exists()))
    tous, bils = [], []
    for d in dossiers:
        try:
            r, b = traiter(d, d.name, a.min_isolats, modeles=tuple(a.modeles),
                           n_temoins=a.temoins)
            if not r.empty:
                tous.append(r); bils.append(b)
        except Exception as e:                                   # noqa: BLE001
            print(f"{d.name} : ECHEC — {type(e).__name__}: {e}")

    if tous:
        res = pd.concat(tous, ignore_index=True)
        bil = pd.concat(bils, ignore_index=True)
        res.to_csv(racine / "loro_tous.csv", index=False)
        bil.to_csv(racine / "loro_regions_tous.csv", index=False)
        resumer(res, bil)
        print(f"\nEcrit : {racine}/loro_tous.csv et loro_regions_tous.csv")


if __name__ == "__main__":
    main()
