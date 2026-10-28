#!/usr/bin/env python3
"""
CABBAGE-Bench — tests statistiques sur les resultats du benchmark.

Repond a une seule question, posee proprement : les ecarts qu'on observe
entre decoupages et entre modeles sont-ils distinguables du bruit ?

CE QUE CE SCRIPT FAIT
  - Wilcoxon apparie sur les antibiotiques, qui sont l'unite d'observation.
    Apparie parce que chaque antibiotique est evalue sous les deux conditions
    comparees ; non parametrique parce que les AUROC ne sont ni normales ni
    homoscedastiques d'un antibiotique a l'autre.
  - Estimateur de Hodges-Lehmann : la mediane des differences appariees.
    C'est la taille d'effet qui correspond au test de Wilcoxon.
  - Intervalle de confiance par bootstrap sur les antibiotiques (10 000 tirages).
  - Correction de Benjamini-Hochberg sur l'ensemble de la famille de tests.

CE QUE CE SCRIPT NE PEUT PAS FAIRE, ET POURQUOI
  Le test de DeLong compare deux AUROC calculees sur les MEMES echantillons et
  exige les scores predits individuels. Le fichier de resultats ne contient que
  des AUROC agregees par antibiotique. DeLong demande donc un second passage ou
  `evaluer()` conserve les predictions. Tant qu'on ne l'a pas, on ne peut pas
  affirmer qu'une AUROC de 0,913 differe significativement d'une AUROC de 0,785
  POUR UN ANTIBIOTIQUE DONNE ; on peut seulement affirmer que la degradation
  est systematique SUR L'ENSEMBLE des antibiotiques, ce qui est d'ailleurs la
  question qui nous interesse.

Usage :
    python stats_benchmark.py resultats_kp.csv
    python stats_benchmark.py bench/resultats_tous.csv --sortie stats/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GRAINE = 0
B = 10000


def hodges_lehmann(d):
    """Mediane des moyennes de toutes les paires : l'estimateur apparie a Wilcoxon."""
    d = np.asarray(d, dtype=float)
    w = (d[:, None] + d[None, :]) / 2.0
    return float(np.median(w[np.triu_indices_from(w)]))


def ic_bootstrap(d, alpha=0.05, b=B, graine=GRAINE):
    """IC de la mediane des differences, par tirage avec remise sur les antibiotiques."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(graine)
    tirages = rng.choice(d, size=(b, d.size), replace=True)
    meds = np.median(tirages, axis=1)
    return (float(np.quantile(meds, alpha / 2)),
            float(np.quantile(meds, 1 - alpha / 2)))


def benjamini_hochberg(p):
    """Renvoie les p corrigees, dans l'ordre d'entree."""
    p = np.asarray(p, dtype=float)
    n = p.size
    ordre = np.argsort(p)
    q = np.empty(n)
    courant = 1.0
    for rang in range(n - 1, -1, -1):
        i = ordre[rang]
        courant = min(courant, p[i] * n / (rang + 1))
        q[i] = courant
    return q


def comparer(large, a, b, metrique, etiquette):
    """Un test apparie entre deux colonnes, sur les antibiotiques communs."""
    if a not in large.columns or b not in large.columns:
        return None
    paire = large[[a, b]].dropna()
    if len(paire) < 6:
        return None
    d = (paire[a] - paire[b]).values
    if np.allclose(d, 0):
        return None
    try:
        stat, p = wilcoxon(paire[a], paire[b], zero_method="wilcox",
                           alternative="two-sided", method="exact"
                           if len(paire) <= 25 else "approx")
    except ValueError:
        return None
    bas, haut = ic_bootstrap(d)
    return {
        "metrique": metrique,
        "comparaison": etiquette,
        "n_antibiotiques": len(paire),
        "moy_a": round(float(paire[a].mean()), 4),
        "moy_b": round(float(paire[b].mean()), 4),
        "diff_moyenne": round(float(d.mean()), 4),
        "hodges_lehmann": round(hodges_lehmann(d), 4),
        "ic95_bas": round(bas, 4),
        "ic95_haut": round(haut, 4),
        "n_degrade": int((d < 0).sum()),
        "n_ameliore": int((d > 0).sum()),
        "p": p,
    }


def analyser(res, espece=""):
    lignes = []
    decoupages = [d for d in ("aleatoire", "lignee", "region", "periode")
                  if d in set(res["decoupage"])]
    modeles = [m for m in ("separe", "multi_label") if m in set(res["modele"])]

    for metrique in ("auroc", "auprc", "err_tres_majeure_pct", "ece"):
        if metrique not in res.columns:
            continue
        large = res.pivot_table(index="antibiotique",
                               columns=["decoupage", "modele"],
                               values=metrique, aggfunc="mean")
        large.columns = [f"{d}|{m}" for d, m in large.columns]

        # 1. Effet du decoupage, a modele fixe : chaque decoupage contre l'aleatoire.
        for m in modeles:
            for d in decoupages:
                if d == "aleatoire":
                    continue
                r = comparer(large, f"{d}|{m}", f"aleatoire|{m}", metrique,
                             f"{d} - aleatoire  [{m}]")
                if r:
                    lignes.append(r)

        # 2. Effet du modele, a decoupage fixe.
        if len(modeles) == 2:
            for d in decoupages:
                r = comparer(large, f"{d}|multi_label", f"{d}|separe", metrique,
                             f"multi_label - separe  [{d}]")
                if r:
                    lignes.append(r)

    t = pd.DataFrame(lignes)
    if t.empty:
        return t
    t["p_bh"] = benjamini_hochberg(t["p"].values)
    t["signif"] = np.where(t["p_bh"] < 0.001, "***",
                  np.where(t["p_bh"] < 0.01, "**",
                  np.where(t["p_bh"] < 0.05, "*", "ns")))
    t["p"] = t["p"].map(lambda v: f"{v:.2e}")
    t["p_bh"] = t["p_bh"].map(lambda v: f"{v:.2e}")
    if espece:
        t.insert(0, "espece", espece)
    return t


def table_latex(t, metrique="auroc"):
    s = t[t["metrique"] == metrique]
    if s.empty:
        return ""
    out = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Comparison & $n$ & Median diff. & 95\% CI & $q$ \\",
           r"\midrule"]
    for _, r in s.iterrows():
        comp = r["comparaison"].replace("_", r"\_")
        out.append(f"{comp} & {r['n_antibiotiques']} & "
                   f"{r['hodges_lehmann']:+.3f} & "
                   f"[{r['ic95_bas']:+.3f}, {r['ic95_haut']:+.3f}] & "
                   f"{r['p_bh']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fichier")
    p.add_argument("--sortie", default=".")
    a = p.parse_args()

    res = pd.read_csv(a.fichier)
    Path(a.sortie).mkdir(parents=True, exist_ok=True)

    morceaux = []
    if "espece" in res.columns and res["espece"].nunique() > 1:
        for esp, sous in res.groupby("espece"):
            morceaux.append(analyser(sous, esp))
    else:
        esp = res["espece"].iloc[0] if "espece" in res.columns else ""
        morceaux.append(analyser(res, esp))

    t = pd.concat([m for m in morceaux if not m.empty], ignore_index=True)
    t.to_csv(Path(a.sortie) / "stats.csv", index=False)

    cols = ["comparaison", "n_antibiotiques", "moy_a", "moy_b",
            "hodges_lehmann", "ic95_bas", "ic95_haut",
            "n_degrade", "n_ameliore", "p_bh", "signif"]
    for metrique in t["metrique"].unique():
        print(f"\n=== {metrique} ===")
        print(t[t["metrique"] == metrique][cols].to_string(index=False))

    (Path(a.sortie) / "table_auroc.tex").write_text(table_latex(t, "auroc"))
    print(f"\nEcrit dans {a.sortie}/stats.csv et table_auroc.tex")


if __name__ == "__main__":
    main()
