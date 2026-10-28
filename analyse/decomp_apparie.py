import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

SRC = Path(os.environ.get("CABBAGE_RESULTS", "resultats")) / "decomposition_tous.csv"
OUT = Path(os.environ.get("CABBAGE_ANALYSIS_OUTPUT", "analysis_output")); OUT.mkdir(parents=True, exist_ok=True)
CLE = ["espece", "analyse", "colonne", "strate", "tenu", "antibiotique", "modele"]

d = pd.read_csv(SRC)
d = d[d.n_test >= 30]

# un controle agrege par cellule, puis appariement avec la ligne tenue
tem = d[d.type == "temoin"].groupby(CLE).agg(
    auroc_temoin=("auroc", "mean"), vme_temoin=("err_tres_majeure_pct", "mean"),
    n_temoins=("auroc", "size")).reset_index()
ten = d[d.type == "tenu"].groupby(CLE).agg(
    auroc_tenu=("auroc", "mean"), vme_tenu=("err_tres_majeure_pct", "mean")).reset_index()
p = ten.merge(tem, on=CLE, how="inner")
p["delta"] = p.auroc_tenu - p.auroc_temoin
p["delta_vme"] = p.vme_tenu - p.vme_temoin
print("cellules appariees :", len(p), " (>=1 temoin)")
print(p.n_temoins.describe()[["min", "mean", "max"]].to_dict())

def bootstrap_grappe(sous, grappe, B=5000, graine=0):
    """bootstrap par grappe : on retire des grappes entieres, pas des cellules."""
    g = sous.groupby(grappe)["delta"].mean()
    noms = g.index.to_numpy(); vals = g.to_numpy()
    rng = np.random.default_rng(graine)
    tir = rng.integers(0, len(vals), size=(B, len(vals)))
    moy = vals[tir].mean(axis=1)
    return vals.mean(), np.percentile(moy, [2.5, 97.5]), len(noms)

print("\n=== decomposition croisee : effet par analyse (toutes colonnes) ===")
lignes = []
for (an, mo), s in p.groupby(["analyse", "modele"]):
    m, ci, ng = bootstrap_grappe(s, ["espece", "colonne", "strate", "tenu"])
    w = wilcoxon(s.auroc_tenu, s.auroc_temoin)
    deg = 100 * (s.delta < 0).mean()
    lignes.append(dict(analyse=an, modele=mo, n_cellules=len(s), n_grappes=ng,
                       delta=round(m, 4), ic_bas=round(ci[0], 4), ic_haut=round(ci[1], 4),
                       degradent_pct=round(deg, 1), p_wilcoxon=w.pvalue,
                       delta_vme=round(s.delta_vme.mean(), 2)))
res = pd.DataFrame(lignes)
print(res.to_string(index=False))
res.to_csv(OUT / "decomp_global.csv", index=False)

print("\n=== par colonne de provenance ===")
lignes = []
for (col, an, mo), s in p.groupby(["colonne", "analyse", "modele"]):
    if len(s) < 20: continue
    m, ci, ng = bootstrap_grappe(s, ["espece", "strate", "tenu"])
    w = wilcoxon(s.auroc_tenu, s.auroc_temoin)
    lignes.append(dict(colonne=col, analyse=an, modele=mo, n=len(s), n_grappes=ng,
                       delta=round(m, 4), ic_bas=round(ci[0], 4), ic_haut=round(ci[1], 4),
                       degradent_pct=round(100 * (s.delta < 0).mean(), 1), p=w.pvalue))
pc = pd.DataFrame(lignes).sort_values(["colonne", "analyse", "modele"])
print(pc.to_string(index=False))
pc.to_csv(OUT / "decomp_colonne.csv", index=False)

print("\n=== par espece (gb) ===")
s0 = p[p.modele == "gb"]
lignes = []
for (esp, an), s in s0.groupby(["espece", "analyse"]):
    if len(s) < 10: continue
    m, ci, ng = bootstrap_grappe(s, ["colonne", "strate", "tenu"])
    lignes.append(dict(espece=esp, analyse=an, n=len(s), delta=round(m, 4),
                       ic_bas=round(ci[0], 4), ic_haut=round(ci[1], 4)))
pe = pd.DataFrame(lignes)
print(pe.to_string(index=False))
pe.to_csv(OUT / "decomp_espece.csv", index=False)

# test apparie direct geographie vs provenance, sur les cellules comparables
print("\n=== geographie vs provenance, apparie par (espece, colonne) ===")
# The primary unit is (species, provenance column). Model families are repeated
# measurements of the same 22 units and are averaged before the paired test.
piv = p.groupby(["espece", "colonne", "analyse"])["delta"].mean().unstack("analyse")
piv = piv.dropna()
print(piv.round(4).to_string())
if len(piv) > 2:
    w = wilcoxon(piv.geographie, piv.provenance)
    print("\ngeographie %.4f vs provenance %.4f | Wilcoxon p=%.4g  n=%d paires"
          % (piv.geographie.mean(), piv.provenance.mean(), w.pvalue, len(piv)))
    print("rapport des effets : %.2fx" % (piv.geographie.mean() / piv.provenance.mean()))
