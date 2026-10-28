#!/usr/bin/env python3
"""La chute geographique est-elle de la geographie, ou de la mesure ?

CE QUE CE SCRIPT AJOUTE A L'ARTICLE
  L'article etablit que tenir une region a l'ecart degrade les performances
  dans 81 % des paires region-antibiotique, et ecarte quatre explications.
  Il n'en ecarte pas une cinquieme : dans une base qui agrege neuf
  ressources, la region est confondue avec qui a depose l'isolat, avec quelle
  methode d'antibiogramme, et sous quelle norme de seuils.

  Le test de faisabilite (provenance.py) a montre que la question est
  attaquable : `database` est couverte a 100 % avec un V de Cramer de 0,27 a
  0,43 contre la region, et `laboratory_typing_method` et `ast_standard`
  existent aussi. Ce script fait la mesure.

TROIS ANALYSES, DE LA MOINS CHERE A LA PLUS CHERE

  A. LE DESACCORD D'ETIQUETTES, SANS AUCUN MODELE.
     CABBAGE fournit Updated_phenotype_CLSI et Updated_phenotype_EUCAST : le
     meme isolat, le meme antibiotique, deux normes de seuils. On compte
     combien d'etiquettes basculent. Ce chiffre borne par le haut tout ce
     qu'un modele peut apprendre, et il ne depend d'aucun choix de notre
     part. A notre connaissance il n'a jamais ete publie sur ce jeu.

  B. LA DECOMPOSITION CROISEE.
     (a) a REGION FIXE, tenir une provenance a l'ecart : l'effet de
         provenance, geographie neutralisee.
     (b) a PROVENANCE FIXE, tenir une region a l'ecart : l'effet de
         geographie, provenance neutralisee.
     Les deux contre un temoin tire au hasard de meme taille dans la meme
     strate, comme le leave-one-region-out de l'article. Si (a) est du meme
     ordre que (b), la « geographie » est pour moitie un effet de lot.

  C. LEAVE-ONE-STUDY-OUT contre LEAVE-ONE-REGION-OUT, globalement.
     Si le premier est aussi severe que le second, c'est lui le protocole
     que le domaine devrait utiliser.

Usage :
    python decomposition.py --bench $BASE/bench --etape A
    python decomposition.py --bench $BASE/bench --etape BC --especes klebsiella_pneumoniae
    python decomposition.py --bench $BASE/bench            # tout
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from baselines_forts import _mesures, charger, colonnes_train

ICI = Path(__file__).resolve().parent
PROV = ("database", "laboratory_typing_method", "ast_standard",
        "isolation_source_category")


def charger_noyau():
    src = (ICI / "cellule_tout.py").read_text(encoding="utf-8")
    src = src.split('print("Pret.")')[0]
    e = {}
    exec(compile(src, "cellule_tout.py", "exec"), e)
    return e


def _con(N):
    con = N["_con"]()
    for pr in ("SET memory_limit='4GB'", "SET threads=4",
               "SET preserve_insertion_order=false"):
        try:
            con.execute(pr)
        except Exception:                                        # noqa: BLE001
            pass
    return con


# ======================================================================
# A. Le desaccord entre normes de seuils, sans modele
# ======================================================================
def etape_A(N, especes, release="2025-11"):
    """Combien de phenotypes changent selon qu'on applique CLSI ou EUCAST ?

    C'est un plafond sur la performance atteignable : si une etiquette sur
    dix depend de la norme choisie, aucun modele ne peut etre juste a mieux
    que cela, et une partie de ce que l'article mesure comme un decalage
    geographique est un desaccord de definition entre laboratoires.
    """
    url = f"{N['FTP']}/{release}/phenotype.parquet"
    tout = []
    for esp in especes:
        esp_sql = esp.replace("_", " ").capitalize()
        con = _con(N)
        try:
            d = con.execute(f"""
                SELECT antibiotic_name AS atb,
                       count(*) AS n,
                       sum(CASE WHEN Updated_phenotype_CLSI
                                  <> Updated_phenotype_EUCAST
                                THEN 1 ELSE 0 END) AS desaccord,
                       count(DISTINCT Updated_phenotype_CLSI) AS n_clsi,
                       count(DISTINCT Updated_phenotype_EUCAST) AS n_eucast
                FROM read_parquet('{url}')
                WHERE species = ?
                  AND Updated_phenotype_CLSI IS NOT NULL
                  AND Updated_phenotype_EUCAST IS NOT NULL
                GROUP BY 1 HAVING count(*) >= 100
                ORDER BY 2 DESC""", [esp_sql]).df()
        except Exception as e:                                   # noqa: BLE001
            print(f"  {esp} : requete impossible — {type(e).__name__}: {e}")
            continue
        finally:
            con.close()
        if d.empty:
            print(f"  {esp} : aucune paire CLSI/EUCAST comparable")
            continue
        d["pct"] = (100 * d.desaccord / d.n).round(2)
        d.insert(0, "espece", esp)
        tout.append(d)
        g = 100 * d.desaccord.sum() / d.n
        print(f"\n=== {esp} : {int(d.n.sum())} paires isolat-antibiotique "
              f"avec les deux normes ===")
        print(f"  desaccord global : {g:.2f} %")
        pires = d.nlargest(5, "pct")[["atb", "n", "desaccord", "pct"]]
        print("  antibiotiques les plus discordants :")
        print("    " + pires.to_string(index=False).replace("\n", "\n    "))
    if not tout:
        return None
    t = pd.concat(tout, ignore_index=True)
    g = 100 * t.desaccord.sum() / t.n.sum()
    print(f"\n{'='*66}")
    print(f"  TOUTES ESPECES : {g:.2f} % des phenotypes changent de valeur")
    print(f"  selon qu'on applique les seuils CLSI ou les seuils EUCAST.")
    print(f"  {int(t.n.sum())} paires isolat-antibiotique, "
          f"{t.atb.nunique()} antibiotiques, {t.espece.nunique()} especes.")
    print(f"{'='*66}")
    return t


# ======================================================================
# B et C. la decomposition croisee
# ======================================================================
def un_modele(nom, Xtr, ytr, Xte, yte, rare):
    if nom == "logistique":
        m = LogisticRegression(max_iter=3000, C=1.0)
    else:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=rare >= 20, validation_fraction=0.15,
            n_iter_no_change=20, random_state=0)
    m.fit(Xtr, ytr)
    return _mesures(m.predict_proba(Xte)[:, 1], yte)


def _paires(X, Y, M, atb, tr, te, modeles, n_min_test, rng, n_temoins=3):
    """Mesure (tenu a l'ecart) contre (temoin aleatoire de meme taille).

    Le temoin est tire DANS LA MEME STRATE, et plusieurs fois : c'est ce qui
    fait que la comparaison isole la variable qu'on fait varier et non la
    taille de l'echantillon ni le bruit d'un tirage unique.
    """
    dispo = np.where(tr | te)[0]
    gard = colonnes_train(X, tr)
    Xr = X[:, gard]
    temoins = []
    for _ in range(n_temoins):
        t_i = np.zeros(X.shape[0], bool)
        t_i[rng.choice(dispo, int(te.sum()), replace=False)] = True
        r_i = (tr | te) & ~t_i
        temoins.append((r_i, t_i, X[:, colonnes_train(X, r_i)]))
    out = []
    for j, nom in enumerate(atb):
        for etiq, mtr, mte, Xu in ([("tenu", tr, te, Xr)]
                                   + [("temoin", a, b, c) for a, b, c in temoins]):
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
                    r = un_modele(mod, Xu[a], Y[a, j], Xu[b], Y[b, j], rare)
                except Exception:                                # noqa: BLE001
                    continue
                out.append({"type": etiq, "modele": mod, "antibiotique": nom,
                            "n_test": int(b.sum()), **r})
    return out


def etape_BC(dossier, espece, N, prov_src, modeles, n_min=30, n_min_test=20):
    X, Y, M, splits, atb, genes = charger(dossier)
    with open(Path(dossier) / "jeu.pkl", "rb") as f:
        meta = pickle.load(f)["meta"]
    Xd = X.toarray().astype(np.float32)
    reg = meta["region"].astype("object")
    rng = np.random.default_rng(0)
    print(f"\n{'='*70}\n=== {espece} : {len(meta)} isolats ===\n{'='*70}")

    lignes = []
    for col in PROV:
        if col not in prov_src.columns:
            continue
        p = prov_src[col].reindex(meta.index).astype("object")
        p = p.where(p.notna() & (p.astype(str).str.strip() != ""), None)
        ok = reg.notna() & p.notna()
        if ok.sum() < 200:
            continue
        t = pd.crosstab(reg[ok], p[ok].astype(str))

        # --- (a) a REGION FIXE, tenir une provenance a l'ecart ----------
        for region in t.index:
            niv = [c for c in t.columns if t.loc[region, c] >= n_min]
            if len(niv) < 2:
                continue
            dans = (reg == region).fillna(False).to_numpy() & ok.to_numpy()
            for c in niv:
                te = dans & (p.astype(str) == c).to_numpy()
                tr = dans & ~te & p.astype(str).isin(niv).to_numpy()
                if te.sum() < 50 or tr.sum() < 150:
                    continue
                for r in _paires(Xd, Y, M, atb, tr, te, modeles,
                                 n_min_test, rng):
                    r.update({"espece": espece, "analyse": "provenance",
                              "colonne": col, "strate": str(region),
                              "tenu": str(c)})
                    lignes.append(r)

        # --- (b) a PROVENANCE FIXE, tenir une region a l'ecart ----------
        for c in t.columns:
            regs = [r_ for r_ in t.index if t.loc[r_, c] >= n_min]
            if len(regs) < 2:
                continue
            dans = (p.astype(str) == c).to_numpy() & ok.to_numpy()
            for region in regs:
                te = dans & (reg == region).fillna(False).to_numpy()
                tr = dans & ~te & reg.isin(regs).fillna(False).to_numpy()
                if te.sum() < 50 or tr.sum() < 150:
                    continue
                for r in _paires(Xd, Y, M, atb, tr, te, modeles,
                                 n_min_test, rng):
                    r.update({"espece": espece, "analyse": "geographie",
                              "colonne": col, "strate": str(c),
                              "tenu": str(region)})
                    lignes.append(r)

        n_a = sum(1 for l in lignes if l["colonne"] == col
                  and l["analyse"] == "provenance" and l["type"] == "tenu")
        n_b = sum(1 for l in lignes if l["colonne"] == col
                  and l["analyse"] == "geographie" and l["type"] == "tenu")
        print(f"  {col:<28} provenance a region fixe : {n_a:4d} mesures   "
              f"geographie a provenance fixe : {n_b:4d}")
    return pd.DataFrame(lignes)


def resumer(res):
    if res.empty:
        print("\nAucune mesure exploitable.")
        return
    cle = ["espece", "analyse", "colonne", "strate", "tenu", "modele",
           "antibiotique"]
    g = res.pivot_table(index=cle, columns="type", values="auroc",
                        aggfunc="mean")
    # Le pivot cree des colonnes "tenu" et "temoin", or "tenu" est deja un
    # niveau d'index : reset_index() refuserait de les faire coexister.
    g = g.rename(columns={"tenu": "auroc_tenu", "temoin": "auroc_temoin"})
    if "auroc_tenu" not in g or "auroc_temoin" not in g:
        print("\nPas de paires tenu/temoin comparables.")
        return
    g = g.dropna(subset=["auroc_tenu", "auroc_temoin"])
    g["chute"] = g["auroc_tenu"] - g["auroc_temoin"]
    e = g.reset_index()

    print(f"\n{'='*70}\n=== LA DECOMPOSITION ===\n{'='*70}")
    rng = np.random.default_rng(0)
    for (an, mod), s in e.groupby(["analyse", "modele"]):
        x = s.chute.values
        bs = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(5000)]
        print(f"  {an:11s} {mod:11s} n={len(x):4d}  "
              f"degradent {100*(x<0).mean():4.1f} %   "
              f"mediane {np.median(x):+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  "
              f"p={stats.wilcoxon(x).pvalue:.1e}")

    print(f"\n--- par colonne de provenance ---")
    print(e.groupby(["colonne", "analyse", "modele"])["chute"]
           .agg(["count", "median"]).round(4).to_string())

    print(f"""
  COMMENT LIRE.

  'provenance' est la chute quand on tient a l'ecart une provenance A
  REGION FIXE : la geographie est neutralisee, il ne reste que l'effet de
  lot, de methode ou de norme.

  'geographie' est la chute quand on tient a l'ecart une region A
  PROVENANCE FIXE : l'effet de lot est neutralise, il ne reste que la
  geographie.

  Si les deux medianes sont du meme ordre, alors ce que l'article mesure
  comme un decalage geographique est pour moitie un decalage de mesure, et
  le protocole du domaine est mal choisi. Si 'geographie' domine largement,
  l'article 1 est renforce : le dernier facteur de confusion evident est
  ecarte.""")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    p.add_argument("--etape", default="ABC", choices=["A", "BC", "ABC"])
    p.add_argument("--modeles", nargs="+", default=["logistique", "gb"])
    a = p.parse_args()

    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "jeu.pkl").exists()))
    especes = [d.name for d in dossiers]
    N = charger_noyau()

    if "A" in a.etape:
        print(f"\n{'#'*70}\n# A. desaccord entre normes de seuils, sans modele"
              f"\n{'#'*70}")
        t = etape_A(N, especes)
        if t is not None:
            t.to_csv(racine / "desaccord_normes.csv", index=False)
            print(f"\nEcrit : {racine}/desaccord_normes.csv")

    if "BC" in a.etape:
        print(f"\n{'#'*70}\n# B. decomposition croisee\n{'#'*70}")
        url = f"{N['FTP']}/2025-11/phenotype.parquet"
        cols = ", ".join(f'any_value("{c}") AS "{c}"' for c in PROV)
        tous = []
        for d in dossiers:
            esp_sql = d.name.replace("_", " ").capitalize()
            con = _con(N)
            try:
                src = con.execute(f"""
                    SELECT BioSample_ID, {cols} FROM read_parquet('{url}')
                    WHERE species = ? AND BioSample_ID IS NOT NULL
                    GROUP BY BioSample_ID""", [esp_sql]).df().set_index("BioSample_ID")
            except Exception as e:                               # noqa: BLE001
                print(f"{d.name} : provenance illisible — {type(e).__name__}")
                continue
            finally:
                con.close()
            try:
                r = etape_BC(d, d.name, N, src, tuple(a.modeles))
                if not r.empty:
                    tous.append(r)
            except Exception as e:                               # noqa: BLE001
                print(f"{d.name} : ECHEC — {type(e).__name__}: {e}")
        if tous:
            res = pd.concat(tous, ignore_index=True)
            res.to_csv(racine / "decomposition_tous.csv", index=False)
            resumer(res)
            print(f"\nEcrit : {racine}/decomposition_tous.csv")


if __name__ == "__main__":
    main()
