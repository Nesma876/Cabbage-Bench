#!/usr/bin/env python3
"""
CABBAGE-Bench — modeles de reference FORTS.

POURQUOI CE SCRIPT EXISTE
  Le benchmark compare pour l'instant une regression logistique et un petit
  goulot partage. Un relecteur objectera, a juste titre, que la domination du
  decalage temporel pourrait etre un artefact de modeles faibles : un modele
  plus capable apprendrait peut-etre les signaux stables et souffrirait moins.
  Tant que l'effet n'est pas verifie sur un modele fort, le resultat central
  de l'article est attaquable.

  On ajoute donc deux references que le domaine considere comme les bonnes :

    regles   Predicteur par regles, dans l'esprit de ResFinder et
             d'AMRFinderPlus : un isolat est declare resistant a un
             antibiotique s'il porte au moins un gene connu de la classe
             correspondante. Transparent, sans apprentissage, et c'est la
             reference clinique reelle.

    rf       Foret aleatoire, un modele par antibiotique. Tres presente dans
             la litterature AMR, et robuste sans reglage : c'est la reference
             que la plupart des relecteurs du domaine auront en tete.

    gb       Gradient boosting par histogrammes, un modele par antibiotique.
             C'est la reference forte etablie : un preprint de 2026 montre
             qu'il depasse les plongements de modeles de fondation sur ce type
             de tache, et que la taille du jeu de donnees compte plus que
             l'architecture.

    logistique  Recalculee ici pour que les trois soient compares sur des
                decoupages strictement identiques.

CE SCRIPT NE MODIFIE RIEN
  Il lit les fichiers produits par run_all.py et ecrit a cote. On peut donc le
  lancer pendant que d'autres especes sont encore en file, sans risque.

Usage :
    python baselines_forts.py --bench $BASE/bench
    python baselines_forts.py --bench $BASE/bench --especes klebsiella_pneumoniae
"""

import argparse
import pickle
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Correspondance classe d'antibiotique -> prefixes de genes de resistance.
# Etablie d'apres la nomenclature du Reference Gene Catalog du NCBI.
# Volontairement conservatrice : mieux vaut un predicteur par regles qui
# manque un mecanisme rare qu'un predicteur qui en invente.
# ---------------------------------------------------------------------------
CLASSES = {
    "beta_lactamines": (
        r"(penicillin|ampicillin|amoxicillin|piperacillin|ticarcillin|oxacillin|"
        r"cefal|cefaz|cefot|ceftaz|ceftri|cefep|cefox|cefur|cefpod|cefix|"
        r"meropenem|imipenem|ertapenem|doripenem|aztreonam|temocillin)",
        ("bla", "mec", "amp", "ges", "imp", "kpc", "ndm", "oxa", "vim", "ctx",
         "shv", "tem", "cmy", "dha", "fox", "mir", "act", "pdc", "ade"),
    ),
    "aminoglycosides": (
        r"(gentamicin|amikacin|tobramycin|kanamycin|streptomycin|netilmicin|"
        r"plazomicin|spectinomycin|apramycin)",
        ("aac", "aph", "ant", "aad", "str", "arm", "rmt", "npm"),
    ),
    "quinolones": (
        r"(ciprofloxacin|levofloxacin|moxifloxacin|norfloxacin|ofloxacin|"
        r"nalidixic|enrofloxacin|gatifloxacin|delafloxacin)",
        ("qnr", "qep", "oqx", "crp"),
    ),
    "tetracyclines": (
        r"(tetracycline|doxycycline|minocycline|tigecycline|eravacycline|"
        r"omadacycline|chlortetracycline)",
        ("tet", "otr"),
    ),
    "sulfamides_trimethoprime": (
        r"(sulfamethoxazole|sulfisoxazole|trimethoprim|sulfadimethoxine|"
        r"cotrimoxazole)",
        ("sul", "dfr", "dhf"),
    ),
    "macrolides": (
        r"(erythromycin|azithromycin|clarithromycin|clindamycin|telithromycin|"
        r"tylosin|tilmicosin|spiramycin)",
        ("erm", "mph", "mef", "msr", "ere", "lnu", "vga", "lsa", "mel"),
    ),
    "phenicoles": (
        r"(chloramphenicol|florfenicol|thiamphenicol)",
        ("cat", "cml", "flo", "cmlA", "estD"),
    ),
    "colistine": (
        r"(colistin|polymyxin)",
        ("mcr", "arn", "eptA", "pmr"),
    ),
    "glycopeptides": (
        r"(vancomycin|teicoplanin|telavancin|dalbavancin)",
        ("van",),
    ),
    "rifamycines": (
        r"(rifampicin|rifampin|rifabutin|rifapentine)",
        ("arr", "rph", "rpo"),
    ),
    "fosfomycine": (r"(fosfomycin)", ("fos", "glp", "uhp")),
    "nitrofuranes": (r"(nitrofurantoin|furazolidone)", ("nfs", "oqx")),
    "oxazolidinones": (r"(linezolid|tedizolid)", ("cfr", "opt", "poxt")),
    "lincosamides": (r"(lincomycin|pirlimycin)", ("lnu", "lsa", "vga")),
}


def classe_de(atb):
    a = str(atb).lower()
    for nom, (motif, _) in CLASSES.items():
        if re.search(motif, a):
            return nom
    return None


# Quelles colonnes de X appartiennent au catalogue AMR. Le baseline a regles
# ne doit lire que celles-la : les noms des GFF sont generiques (bla_1, bla_2)
# et ne disent pas de quelle beta-lactamase il s'agit. Variable de module
# plutot que septieme valeur de retour, pour ne pas casser geographie.py qui
# importe charger() et depaquete six elements.
MASQUE_AMR = None


def charger(d):
    """Charge la matrice. CABBAGE_MATRICE=pan prend le contenu genique du
    genome entier concatene au bloc catalogue ; sinon le catalogue seul.

    Le catalogue seul etait l'etat initial du benchmark, et c'etait un defaut :
    comparer un modele appris a un systeme de regles sur un vocabulaire fait
    uniquement de genes de resistance revient a comparer deux lectures du meme
    catalogue. Avec la matrice pan, les modeles appris voient le genome et les
    regles voient le catalogue ; la comparaison a enfin un sens.
    """
    global MASQUE_AMR
    d = Path(d)
    quelle = os.environ.get("CABBAGE_MATRICE", "amr").lower()
    if quelle != "amr":
        # `pan` = contenu genique des noms de GFF (refaire_matrice.py)
        # `pg`  = familles d'orthologues MMseqs2 (pangenome_matrice.py)
        f = d / f"X_genes_{quelle}.npz"
        if not f.exists():
            raise SystemExit(f"{f} absent : construire cette matrice d'abord")
        X = sparse.load_npz(f)
        v = pd.read_csv(d / f"genes_{quelle}.csv")
        MASQUE_AMR = (v["bloc"] == "amr").to_numpy()
        print(f"  matrice {quelle} : {X.shape[1]} caracteristiques "
              f"({int(MASQUE_AMR.sum())} du catalogue AMR, "
              f"{int((~MASQUE_AMR).sum())} du contenu genique)")
        Y = np.load(d / "Y.npy"); M = np.load(d / "M.npy")
        splits = pd.read_csv(d / "splits.csv", index_col=0)
        atb = pd.read_csv(d / "antibiotiques.csv")["antibiotique"].tolist()
        return X, Y, M, splits, atb, v["gene"].tolist()
    MASQUE_AMR = None
    X = sparse.load_npz(d / "X_genes.npz")
    Y = np.load(d / "Y.npy")
    M = np.load(d / "M.npy")
    splits = pd.read_csv(d / "splits.csv", index_col=0)
    atb = pd.read_csv(d / "antibiotiques.csv")["antibiotique"].tolist()
    # The release deliberately omits the construction-time ``jeu.pkl``.
    # The AMR vocabulary is preserved as the AMR block of genes_pan.csv.
    vocab = pd.read_csv(d / "genes_pan.csv")
    genes = vocab.loc[vocab["bloc"] == "amr", "gene"].tolist()
    if len(genes) != X.shape[1]:
        raise ValueError(f"AMR vocabulary has {len(genes)} rows for {X.shape[1]} columns")
    return X, Y, M, splits, atb, genes


def _ece(p, y, n=10):
    """Erreur de calibration attendue, 10 intervalles d'egale largeur."""
    bords = np.linspace(0, 1, n + 1)
    e = 0.0
    for i in range(n):
        s = (p >= bords[i]) & (p < bords[i + 1] if i < n - 1 else p <= 1.0)
        if s.sum():
            e += s.sum() / len(p) * abs(y[s].mean() - p[s].mean())
    return e


def _mesures(p, y):
    pred = p >= 0.5
    r, s = y == 1, y == 0
    return {
        "auroc": round(roc_auc_score(y, p), 4),
        "auprc": round(average_precision_score(y, p), 4),
        "err_majeure_pct": round(100 * (s & pred).sum() / max(s.sum(), 1), 1),
        "err_tres_majeure_pct": round(100 * (r & ~pred).sum() / max(r.sum(), 1), 1),
        "ece": round(_ece(p, y), 4),
    }


def colonnes_train(X, tr, min_prev=0.01, max_prev=0.99, min_n=10):
    """Colonnes retenues en ne regardant QUE les lignes d'entrainement.

    refaire_matrice.py et pangenome_matrice.py construisaient la bande de
    prevalence sur TOUS les genomes, avant tout decoupage. Pour le protocole
    geographique, le choix des caracteristiques connaissait donc le contenu
    genique de la region tenue a l'ecart : une fuite, modeste mais reelle,
    dans un article dont le sujet est la fuite. On refait le filtre ici,
    protocole par protocole, sur les seules lignes d'entrainement.

    Le bloc du catalogue AMR est toujours conserve : il est fixe, defini par
    une nomenclature externe, et c'est lui que lit le predicteur par regles.
    Le filtrer reviendrait a changer le comparateur d'un protocole a l'autre.
    """
    c = np.asarray(X[tr].sum(axis=0)).ravel()
    n_tr = int(tr.sum())
    garde = (c >= max(min_n, 1)) & (c >= min_prev * n_tr) & (c <= max_prev * n_tr)
    if MASQUE_AMR is not None:
        garde = garde | MASQUE_AMR
    if garde.sum() < 10:                      # garde-fou : filtre trop severe
        garde = np.ones(X.shape[1], bool)
    return garde


def _regles(Xd, genes, atb_nom):
    """Score = nombre de genes de la classe presents. Aucun apprentissage."""
    cl = classe_de(atb_nom)
    if cl is None:
        return None
    prefixes = tuple(p.lower() for p in CLASSES[cl][1])
    col = [i for i, g in enumerate(genes)
           if str(g).lower().startswith(prefixes)
           and (MASQUE_AMR is None or MASQUE_AMR[i])]
    if not col:
        return None
    n = np.asarray(Xd[:, col].sum(axis=1)).ravel()
    # Seuil clinique : ResFinder et AMRFinderPlus appellent la resistance des
    # qu'UN determinant est present. L'ancienne version divisait le compte par
    # 3, si bien que la regle ne declenchait qu'a partir de deux genes et que
    # ses taux d'erreur tres majeure etaient un artefact de ce choix. Un gene
    # donne desormais 0,5 exactement, et le compte ne sert plus qu'a ordonner
    # pour l'AUROC.
    return np.where(n == 0, 0.0, 0.5 + 0.5 * np.minimum(n - 1, 3) / 3.0)


def traiter(dossier, espece, n_min=20):
    X, Y, M, splits, atb, genes = charger(dossier)
    Xd = X.tocsc()
    Xdense = X.toarray().astype(np.float32)
    print(f"\n=== {espece} : {X.shape[0]} isolats, {X.shape[1]} genes, "
          f"{len(atb)} antibiotiques ===")
    sans_classe = [a for a in atb if classe_de(a) is None]
    if sans_classe:
        print(f"    {len(sans_classe)} antibiotiques sans classe reconnue "
              f"(pas de predicteur par regles) : {', '.join(sans_classe[:6])}"
              + (" ..." if len(sans_classe) > 6 else ""))

    lignes, echecs = [], []
    for dec in [c for c in ("aleatoire", "lignee", "region", "periode")
                if c in splits.columns]:
        # Les colonnes de decoupage contiennent les CHAINES "train", "val" et
        # "test", pas des booleens. Un .astype(bool) sur une chaine non vide
        # renvoie True : tout basculait en test et l'entrainement etait vide.
        # On reprend la convention exacte de evaluer() : les lignes "val" ne
        # servent ni a l'entrainement ni au test.
        col = splits[dec].astype(str).str.strip().str.lower()
        tr = (col == "train").to_numpy()
        te = (col == "test").to_numpy()
        if te.sum() < 50 or tr.sum() < 200:
            print(f"    {dec} : inutilisable "
                  f"({tr.sum()} en train, {te.sum()} en test ; "
                  f"valeurs vues : {sorted(col.unique())[:5]})")
            continue

        # Vocabulaire choisi sur l'entrainement de CE protocole.
        gard = colonnes_train(X, tr)
        Xp = Xdense[:, gard]
        print(f"    {dec} : {int(gard.sum())} caracteristiques retenues "
              f"sur {X.shape[1]} (vocabulaire d'entrainement)")

        for j, nom in enumerate(atb):
            a = tr & (M[:, j] > 0)
            b = te & (M[:, j] > 0)
            if b.sum() < n_min or len(np.unique(Y[b, j])) < 2:
                continue

            # --- predicteur par regles : aucun entrainement -----------------
            sr = _regles(Xd, genes, nom)
            if sr is not None and len(np.unique(sr[b])) > 1:
                lignes.append({"decoupage": dec, "modele": "regles",
                               "antibiotique": nom, "n": int(b.sum()),
                               "pct_resistant": round(100 * Y[b, j].mean(), 1),
                               **_mesures(sr[b], Y[b, j])})

            if a.sum() < 30 or len(np.unique(Y[a, j])) < 2:
                continue
            # Effectif de la classe la plus rare dans l'entrainement : c'est
            # lui qui decide ce qu'on peut entrainer, pas le total.
            rare = int(min((Y[a, j] == 0).sum(), (Y[a, j] == 1).sum()))
            if rare < 5:
                continue                      # trop peu pour apprendre quoi que ce soit

            # --- regression logistique -------------------------------------
            try:
                m = LogisticRegression(max_iter=3000, C=1.0)
                m.fit(Xp[a], Y[a, j])
                lignes.append({"decoupage": dec, "modele": "logistique",
                               "antibiotique": nom, "n": int(b.sum()),
                               "pct_resistant": round(100 * Y[b, j].mean(), 1),
                               **_mesures(m.predict_proba(Xp[b])[:, 1],
                                          Y[b, j])})
            except Exception as e:                               # noqa: BLE001
                echecs.append(f"{dec}/{nom}/logistique: {type(e).__name__}")

            # --- gradient boosting, la reference forte ---------------------
            # L'auto-evaluation preleve 15 % des donnees de facon STRATIFIEE :
            # avec une classe a un seul membre elle est impossible et scikit
            # leve une exception. On ne l'active donc qu'au-dela de 20 isolats
            # dans la classe rare, ce qui laisse de quoi la stratifier.
            try:
                tot = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                    early_stopping=rare >= 20, validation_fraction=0.15,
                    n_iter_no_change=20, random_state=0)
                tot.fit(Xp[a], Y[a, j])
                lignes.append({"decoupage": dec, "modele": "gb",
                               "antibiotique": nom, "n": int(b.sum()),
                               "pct_resistant": round(100 * Y[b, j].mean(), 1),
                               **_mesures(tot.predict_proba(Xp[b])[:, 1],
                                          Y[b, j])})
            except Exception as e:                               # noqa: BLE001
                echecs.append(f"{dec}/{nom}/gb: {type(e).__name__}: {e}")

            # --- foret aleatoire -------------------------------------------
            # Pas d'auto-evaluation interne, donc aucun probleme de classe
            # rare ; elle tolere le desequilibre bien mieux que le boosting.
            try:
                rf = RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                    random_state=0)
                rf.fit(Xp[a], Y[a, j])
                lignes.append({"decoupage": dec, "modele": "rf",
                               "antibiotique": nom, "n": int(b.sum()),
                               "pct_resistant": round(100 * Y[b, j].mean(), 1),
                               **_mesures(rf.predict_proba(Xp[b])[:, 1],
                                          Y[b, j])})
            except Exception as e:                               # noqa: BLE001
                echecs.append(f"{dec}/{nom}/rf: {type(e).__name__}: {e}")

        t = pd.DataFrame([l for l in lignes if l["decoupage"] == dec])
        if not t.empty:
            print("    " + dec.ljust(10) + "  " + "   ".join(
                f"{m} {t[t.modele == m]['auroc'].mean():.4f} "
                f"(n={len(t[t.modele == m])})"
                for m in ("regles", "logistique", "rf", "gb")
                if (t.modele == m).any()))

    if echecs:
        print(f"    {len(echecs)} ajustements en echec, ignores :")
        for e in echecs[:8]:
            print("      -", e)
        if len(echecs) > 8:
            print(f"      ... et {len(echecs)-8} autres")
    res = pd.DataFrame(lignes)
    if not res.empty:
        res.insert(0, "espece", espece)
        res.to_csv(Path(dossier) / "resultats_forts.csv", index=False)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--especes", nargs="*", default=None)
    a = p.parse_args()

    racine = Path(a.bench)
    dossiers = ([racine / e for e in a.especes] if a.especes
                else sorted(d for d in racine.iterdir()
                            if d.is_dir() and (d / "Y.npy").exists()
                            and (d / "jeu.pkl").exists()))
    if not dossiers:
        raise SystemExit(f"Aucune espece complete dans {racine}.")

    tous = []
    for d in dossiers:
        try:
            r = traiter(d, d.name)
            if not r.empty:
                tous.append(r)
        except Exception as e:                                   # noqa: BLE001
            print(f"{d.name} : ECHEC — {type(e).__name__}: {e}")

    if tous:
        t = pd.concat(tous, ignore_index=True)
        t.to_csv(racine / "resultats_forts_tous.csv", index=False)
        print("\n=== AUROC moyen par espece, decoupage et modele ===")
        print(t.pivot_table(index=["espece", "decoupage"], columns="modele",
                            values="auroc", aggfunc="mean").round(3).to_string())
        print("\n=== Erreur tres majeure moyenne (%) ===")
        print(t.pivot_table(index=["espece", "decoupage"], columns="modele",
                            values="err_tres_majeure_pct", aggfunc="mean")
               .round(1).to_string())
        print("\nLA QUESTION QUI DECIDE DE L'ARTICLE :")
        print("l'ecart entre 'aleatoire' et 'periode' subsiste-t-il pour 'gb' ?")
        print("S'il subsiste, le resultat tient pour un modele fort et il est")
        print("solide. S'il disparait, il etait un artefact de modeles faibles")
        print("et c'est l'article qu'il faut repenser, pas le resultat qu'il")
        print("faut defendre.")


if __name__ == "__main__":
    main()
