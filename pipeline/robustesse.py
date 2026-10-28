#!/usr/bin/env python3
"""
Deux controles de robustesse que tout relecteur demandera.

A. GRAINES REPETEES
   Chaque protocole du benchmark repose sur UN tirage. Le decoupage aleatoire
   est une graine ; le decoupage par lignee est un tirage des lignees envoyees
   en test. Une chute de 0,107 mesuree une fois ne dit pas si elle est stable.
   On repete donc avec plusieurs graines et on rapporte la dispersion.

   Les decoupages par region et par periode ne sont pas aleatoires : la region
   de test est choisie par effectif, la coupure temporelle par quantile. Pour
   eux, l'equivalent est de faire varier le CHOIX — chaque region a son tour
   (c'est le leave-one-region-out, deja fait) et plusieurs annees de coupure.

B. RECALIBRATION
   Fait preliminaire, a enoncer avant de lire quoi que ce soit : une
   recalibration est une transformation MONOTONE des scores, donc elle laisse
   l'AUROC inchangee. La chute d'AUROC est une perte de DISCRIMINATION, et
   aucune recalibration ne la repare. Ce n'est pas un resultat experimental,
   c'est une propriete de la mesure.

   En revanche l'erreur de calibration et le taux d'erreurs tres majeures
   dependent du seuil, donc une recalibration peut les ameliorer. On mesure
   deux variantes :

     intra-domaine  Platt ajuste sur une tranche tenue a l'ecart de
                    l'ENTRAINEMENT. C'est ce qu'on peut faire en pratique,
                    puisqu'on n'a pas d'etiquettes du domaine cible.
     oracle         Platt ajuste sur 20 % du TEST, applique aux 80 % restants.
                    Irrealisable en deploiement : c'est une borne superieure.

   La lecture : si l'oracle ramene les erreurs tres majeures a leur niveau
   aleatoire, la degradation etait un probleme de seuil. S'il n'y arrive pas,
   c'est bien la discrimination qui est perdue, et il n'y a pas de remede.

Usage :
    python robustesse.py --bench $BASE/bench --quoi graines
    python robustesse.py --bench $BASE/bench --quoi recalibration
    python robustesse.py --bench $BASE/bench --quoi tout --especes klebsiella_pneumoniae
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ICI = Path(__file__).resolve().parent


def charger_noyau():
    src = (ICI / "cellule_tout.py").read_text(encoding="utf-8")
    src = src.split('print("Pret.")')[0]
    espace = {}
    exec(compile(src, "cellule_tout.py", "exec"), espace)
    return espace


def _ece(p, y, n=10):
    bords = np.linspace(0, 1, n + 1)
    e = 0.0
    for i in range(n):
        s = (p >= bords[i]) & (p < bords[i + 1] if i < n - 1 else p <= 1.0)
        if s.sum():
            e += s.sum() / len(p) * abs(y[s].mean() - p[s].mean())
    return float(e)


def _etm(p, y, seuil=0.5):
    """Erreur tres majeure : un resistant annonce sensible."""
    r = y == 1
    return float((r & (p < seuil)).sum() / max(r.sum(), 1))


def _ajuster(nom, X, y, rare):
    if nom == "logistique":
        m = LogisticRegression(max_iter=3000, C=1.0)
    else:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=rare >= 20, validation_fraction=0.15,
            n_iter_no_change=20, random_state=0)
    m.fit(X, y)
    return m


def charger(d):
    d = Path(d)
    with open(d / "jeu.pkl", "rb") as f:
        jeu = pickle.load(f)
    X = sparse.load_npz(d / "X_genes.npz").toarray().astype(np.float32)
    f_lig = d / "lignees.csv"
    lig = (pd.read_csv(f_lig, index_col=0)["lignee"] if f_lig.exists() else None)
    return jeu, X, lig


# ---------------------------------------------------------------------------
# A. graines repetees
# ---------------------------------------------------------------------------

def graines(dossier, espece, N, n_graines, modele, n_min=20):
    jeu, X, lig = charger(dossier)
    Y, M, meta, atb = jeu["Y"], jeu["M"], jeu["meta"], jeu["antibiotiques"]
    lignes = []
    for g in range(n_graines):
        sp = N["_splits"](meta, graine=g)
        if lig is not None:
            sp["lignee"] = N["splits_par_lignee"](meta, lig, graine=g)
        for dec in ("aleatoire", "lignee"):
            if dec not in sp.columns:
                continue
            col = sp[dec].astype(str)
            tr = (col == "train").to_numpy()
            te = (col == "test").to_numpy()
            if te.sum() < 50 or tr.sum() < 200:
                continue
            vals = []
            for j, nom in enumerate(atb):
                a, b = tr & (M[:, j] > 0), te & (M[:, j] > 0)
                if b.sum() < n_min or len(np.unique(Y[b, j])) < 2: continue
                if a.sum() < 30 or len(np.unique(Y[a, j])) < 2: continue
                rare = int(min((Y[a, j] == 0).sum(), (Y[a, j] == 1).sum()))
                if rare < 5: continue
                try:
                    m = _ajuster(modele, X[a], Y[a, j], rare)
                    vals.append(roc_auc_score(Y[b, j], m.predict_proba(X[b])[:, 1]))
                except Exception:                                # noqa: BLE001
                    continue
            if vals:
                lignes.append({"espece": espece, "graine": g, "decoupage": dec,
                               "modele": modele, "n_antibiotiques": len(vals),
                               "auroc_moyen": round(float(np.mean(vals)), 4)})
        print(f"  graine {g} : " + "  ".join(
            f"{l['decoupage']} {l['auroc_moyen']:.4f}"
            for l in lignes if l["graine"] == g))
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# B. recalibration
# ---------------------------------------------------------------------------

def _platt(s_cal, y_cal, n_min=50, min_classe=10):
    """Regression logistique a une variable sur les scores : Platt scaling.

    Les seuils ne sont pas decoratifs. Ajuste sur vingt points, Platt est plus
    bruyant que le defaut qu'il corrige : sur nos essais il DEGRADAIT la
    calibration. On exige donc 50 points et 10 de la classe minoritaire, et on
    renonce sinon — une recalibration qu'on ne peut pas ajuster proprement
    n'est pas une recalibration.
    """
    if len(y_cal) < n_min:
        return None
    k = int(min((y_cal == 0).sum(), (y_cal == 1).sum()))
    if k < min_classe:
        return None
    m = LogisticRegression(max_iter=1000)
    m.fit(s_cal.reshape(-1, 1), y_cal)
    return lambda s: m.predict_proba(s.reshape(-1, 1))[:, 1]


def recalibration(dossier, espece, modele, n_min=40, graine=0):
    jeu, X, _ = charger(dossier)
    Y, M, atb, sp = jeu["Y"], jeu["M"], jeu["antibiotiques"], jeu["splits"]
    rng = np.random.default_rng(graine)
    lignes = []
    for dec in [c for c in ("aleatoire", "lignee", "region", "periode")
                if c in sp.columns]:
        col = sp[dec].astype(str)
        tr = (col == "train").to_numpy()
        te = (col == "test").to_numpy()
        if te.sum() < 100 or tr.sum() < 200:
            continue
        for j, nom in enumerate(atb):
            a, b = tr & (M[:, j] > 0), te & (M[:, j] > 0)
            if b.sum() < n_min or len(np.unique(Y[b, j])) < 2: continue
            if a.sum() < 60 or len(np.unique(Y[a, j])) < 2: continue
            ia, ib = np.where(a)[0], np.where(b)[0]
            rng.shuffle(ia); rng.shuffle(ib)
            # tranche de calibration PRISE SUR L'ENTRAINEMENT, donc retiree
            # de l'ajustement : sinon les scores de calibration sont ceux
            # d'exemples deja vus, et Platt apprend sur du sur-apprentissage.
            n_cal = max(80, len(ia) // 5)
            cal_in, fit = ia[:n_cal], ia[n_cal:]
            rare = int(min((Y[fit, j] == 0).sum(), (Y[fit, j] == 1).sum()))
            if len(fit) < 60 or rare < 5: continue
            # tranche oracle PRISE SUR LE TEST
            n_or = max(60, len(ib) // 5)
            cal_or, eval_ = ib[:n_or], ib[n_or:]
            if len(eval_) < 60 or len(np.unique(Y[eval_, j])) < 2: continue
            try:
                m = _ajuster(modele, X[fit], Y[fit, j], rare)
            except Exception:                                    # noqa: BLE001
                continue
            s_eval = m.predict_proba(X[eval_])[:, 1]
            y_eval = Y[eval_, j]
            r = {"espece": espece, "decoupage": dec, "modele": modele,
                 "antibiotique": nom, "n_eval": len(eval_),
                 "auroc": round(roc_auc_score(y_eval, s_eval), 4),
                 "ece_brut": round(_ece(s_eval, y_eval), 4),
                 "etm_brut_pct": round(100 * _etm(s_eval, y_eval), 1)}
            f_in = _platt(m.predict_proba(X[cal_in])[:, 1], Y[cal_in, j])
            if f_in is not None:
                p = f_in(s_eval)
                r["ece_intra"] = round(_ece(p, y_eval), 4)
                r["etm_intra_pct"] = round(100 * _etm(p, y_eval), 1)
                # preuve numerique que l'AUROC est invariante
                r["auroc_intra"] = round(roc_auc_score(y_eval, p), 4)
            f_or = _platt(m.predict_proba(X[cal_or])[:, 1], Y[cal_or, j])
            if f_or is not None:
                p = f_or(s_eval)
                r["ece_oracle"] = round(_ece(p, y_eval), 4)
                r["etm_oracle_pct"] = round(100 * _etm(p, y_eval), 1)
            lignes.append(r)
        print(f"  {dec:10s} {sum(1 for l in lignes if l['decoupage']==dec):3d} antibiotiques")
    return pd.DataFrame(lignes)


def resumer_recal(t):
    if t.empty:
        return
    cols = [c for c in ("ece_brut", "ece_intra", "ece_oracle",
                        "etm_brut_pct", "etm_intra_pct", "etm_oracle_pct",
                        "auroc", "auroc_intra") if c in t.columns]
    print("\n=== Moyennes par decoupage ===")
    print(t.groupby("decoupage")[cols].mean().round(3).to_string())
    if {"auroc", "auroc_intra"} <= set(t.columns):
        d = (t["auroc"] - t["auroc_intra"]).abs().max()
        print(f"\nEcart maximal d'AUROC avant/apres recalibration : {d:.2e}")
        print("  (il doit etre nul au bruit numerique pres : une")
        print("   transformation monotone ne change pas l'AUROC)")
    if {"etm_brut_pct", "etm_oracle_pct"} <= set(t.columns):
        g = t.groupby("decoupage")[["etm_brut_pct", "etm_oracle_pct"]].mean()
        if "aleatoire" in g.index:
            base = g.loc["aleatoire", "etm_brut_pct"]
            print("\n=== Part de la hausse d'erreurs tres majeures que "
                  "l'oracle efface ===")
            for dec in g.index:
                if dec == "aleatoire": continue
                hausse = g.loc[dec, "etm_brut_pct"] - base
                reste = g.loc[dec, "etm_oracle_pct"] - base
                if hausse <= 0.5:
                    continue
                msg = (f"  {dec:10s} hausse {hausse:+5.1f} pts  ->  "
                       f"{reste:+5.1f} apres oracle")
                if reste > hausse:
                    msg += "   (l'oracle AGGRAVE : tranche trop petite ?)"
                else:
                    msg += f"   ({100*(1-reste/hausse):3.0f} % efface)"
                print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    p.add_argument("--n-min-eval", type=int, default=40,
                   help="antibiotiques trop peu testes : ignores")
    p.add_argument("--quoi", choices=["graines", "recalibration", "tout"],
                   default="tout")
    p.add_argument("--graines", type=int, default=10)
    p.add_argument("--modele", default="logistique",
                   choices=["logistique", "gb"])
    a = p.parse_args()

    N = charger_noyau()
    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "jeu.pkl").exists()))

    if a.quoi in ("graines", "tout"):
        tous = []
        for d in dossiers:
            print(f"\n=== graines : {d.name} ===")
            try:
                t = graines(d, d.name, N, a.graines, a.modele)
                if not t.empty:
                    t.to_csv(d / "graines.csv", index=False); tous.append(t)
            except Exception as e:                               # noqa: BLE001
                print(f"  ECHEC — {type(e).__name__}: {e}")
        if tous:
            g = pd.concat(tous, ignore_index=True)
            g.to_csv(racine / "graines_tous.csv", index=False)
            print("\n=== AUROC moyen sur les graines : moyenne, ecart-type, "
                  "etendue ===")
            r = g.groupby(["espece", "decoupage"])["auroc_moyen"].agg(
                ["count", "mean", "std", "min", "max"]).round(4)
            r["etendue"] = (r["max"] - r["min"]).round(4)
            print(r.to_string())
            print("\nUn ecart-type faible devant l'effet mesure signifie que")
            print("la chute rapportee ne depend pas du tirage.")

    if a.quoi in ("recalibration", "tout"):
        tous = []
        for d in dossiers:
            print(f"\n=== recalibration : {d.name} ===")
            try:
                t = recalibration(d, d.name, a.modele)
                if not t.empty:
                    t.to_csv(d / "recalibration.csv", index=False); tous.append(t)
            except Exception as e:                               # noqa: BLE001
                print(f"  ECHEC — {type(e).__name__}: {e}")
        if tous:
            g = pd.concat(tous, ignore_index=True)
            g.to_csv(racine / "recalibration_tous.csv", index=False)
            resumer_recal(g)


if __name__ == "__main__":
    main()
