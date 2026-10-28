"""Le controle 'nouveaute de lignee', refait proprement dans les trois
representations. C'est le paragraphe que l'audit a declare fatal : il etait
calcule sur un run perime et rapporte dans la seule representation ou il est
nul."""
import os
from pathlib import Path
import pandas as pd, numpy as np, itertools
from scipy.stats import pearsonr, spearmanr

P = Path(os.environ.get("CABBAGE_RESULTS", "resultats"))
reg = pd.read_csv(P / "amr_loro_regions_tous.csv").set_index(["espece", "region"])
CLE = ["espece", "region", "modele", "antibiotique"]

def chutes(k, seuil=30):
    s = pd.read_csv(P / f"{k}_loro_tous.csv")
    s = s[s.n_test >= seuil]
    ten = s[s.type == "region"].groupby(CLE).auroc.mean().rename("t")
    tem = s[s.type == "temoin"].groupby(CLE).auroc.mean().rename("c")
    m = pd.concat([ten, tem], axis=1).dropna()
    m["d"] = m.t - m.c
    return m.reset_index()

D = {k: chutes(k) for k in ["amr", "pan", "pg"]}

print("=== correlation chute(region) ~ %% de lignees deja vues, n=21 ===")
for k in D:
    for mo in sorted(D[k].modele.unique()):
        g = D[k][D[k].modele == mo].groupby(["espece", "region"]).d.median()
        x = reg.lignees_vues_pct.reindex(g.index)
        r, p = pearsonr(x, g); rs, ps = spearmanr(x, g)
        print("  %-4s %-11s r=%+.3f p=%.4f   rho=%+.3f p=%.4f" % (k, mo, r, p, rs, ps))

print("\n=== les deux regions les plus nouvelles ===")
print(reg.lignees_vues_pct.sort_values().head(4).to_string())
for k in D:
    g = D[k][D[k].modele == "gb"].groupby(["espece", "region"]).d.median()
    print("  %s : chute de ces quatre regions %s" %
          (k, g.reindex(reg.lignees_vues_pct.sort_values().head(4).index).round(3).to_dict()))

print("\n=== regions a familiarite elevee (>= 94%%) ===")
haut = reg[reg.lignees_vues_pct >= 94].index
print("  n =", len(haut))
for k in D:
    for mo in ["gb", "logistique"]:
        g = D[k][D[k].modele == mo].groupby(["espece", "region"]).d.median().reindex(haut).dropna()
        print("  %-4s %-11s %d/%d se degradent, mediane %+.4f" %
              (k, mo, (g < 0).sum(), len(g), g.median()))

print("\n=== quatre especes dont la validation de lignee est la meilleure ===")
TOP = ["escherichia_coli", "acinetobacter_baumannii", "pseudomonas_aeruginosa",
       "salmonella_enterica"]
for k in D:
    for mo in ["gb", "logistique"]:
        g = D[k][(D[k].modele == mo) & (D[k].espece.isin(TOP))].groupby(
            ["espece", "region"]).d.median()
        print("  %-4s %-11s %d/%d se degradent, mediane %+.4f" %
              (k, mo, (g < 0).sum(), len(g), g.median()))

print("\n=== maximum atteignable en excluant deux regions (catalogue) ===")
for mo in ["gb", "logistique"]:
    g = D["amr"][D["amr"].modele == mo].groupby(["espece", "region"]).d.median()
    x = reg.lignees_vues_pct.reindex(g.index)
    best = max((pearsonr(x.drop(list(c)), g.drop(list(c)))[0], c)
               for c in itertools.combinations(g.index, 2))
    print("  %-11s r max = %+.3f en excluant %s" % (mo, best[0], list(best[1])))
