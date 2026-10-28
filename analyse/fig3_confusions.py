#!/usr/bin/env python3
"""Figure 3, refaite : les controles de la section geographique.

L'ancienne version titrait ses deux panneaux « Not a prevalence shift » et
« Not a lineage effect ». Le second est faux : la nullite de la correlation
avec la familiarite des lignees ne vaut que sur le catalogue. Dans les deux
representations plus riches elle est forte et significative. Le panneau de
droite montre donc maintenant les trois representations, et les deux panneaux
utilisent la meme agregation (gradient boosting).
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BLEU, VERT, ORANGE, ROSE = "#0072B2", "#009E73", "#E69F00", "#CC79A7"
ENCRE, ENCRE2, GRILLE = "#1a1a1a", "#595959", "#d9d9d9"
D = Path(os.environ.get("CABBAGE_RESULTS", "resultats"))
OUT = Path(os.environ.get("CABBAGE_ANALYSIS_OUTPUT", "analysis_output")); OUT.mkdir(parents=True, exist_ok=True)
SRC = {"Catalogue": D / "amr_loro_tous.csv", "Gene content": D / "pan_loro_tous.csv",
       "Orthologue families": D / "pg_loro_tous.csv"}
CLE = ["espece", "region", "modele", "antibiotique"]

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "axes.edgecolor": ENCRE2,
    "axes.linewidth": 0.6, "axes.labelcolor": ENCRE, "axes.titlesize": 9.5,
    "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.5,
    "xtick.color": ENCRE2, "ytick.color": ENCRE2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def joli(e):
    g, s = e.split("_", 1); return f"$\\it{{{g[0].upper()}.\\ {s}}}$"

def chutes(f):
    s = pd.read_csv(f); s = s[s.n_test >= 30]
    ten = s[s.type == "region"].groupby(CLE).auroc.mean().rename("t")
    tem = s[s.type == "temoin"].groupby(CLE).auroc.mean().rename("c")
    m = pd.concat([ten, tem], axis=1).dropna(); m["chute"] = m.t - m.c
    pr = s[s.type == "region"].groupby(CLE)[["prev_train_pct", "prev_test_pct"]].mean()
    return m.join(pr).reset_index()

reg = pd.read_csv(D / "amr_loro_regions_tous.csv")
A = {k: chutes(f) for k, f in SRC.items()}
pal = [BLEU, VERT, ORANGE, ROSE, "#56B4E9", "#D55E00"]
formes = ["o", "s", "^", "D", "v", "P"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
fig.subplots_adjust(wspace=0.44, top=0.80)

# --- gauche : prevalence, par antibiotique, catalogue / gradient boosting ---
a = axes[0]
j = A["Catalogue"]; j = j[j.modele == "gb"].copy()
j["d_prev"] = (j.prev_test_pct - j.prev_train_pct).abs()
for k, esp in enumerate(sorted(j.espece.unique())):
    s = j[j.espece == esp]
    a.scatter(s.d_prev, s.chute, s=13, color=pal[k % 6], marker=formes[k % 6],
              alpha=0.6, linewidth=0, label=joli(esp))
a.axhline(0, color=ENCRE, lw=0.8)
r, p = pearsonr(j.d_prev, j.chute)
a.annotate(f"$r = {r:+.3f}$", (0.97, 0.05), xycoords="axes fraction",
           ha="right", fontsize=8, color=ENCRE)
a.set_xlabel("|prevalence shift| (percentage points)")
a.set_ylabel("AUROC difference (per antimicrobial)")
a.set_title("Prevalence shift", loc="left", color=ENCRE, pad=6)

# --- droite : familiarite des lignees, par region, LES TROIS representations ---
b = axes[1]
coul = {"Catalogue": ENCRE2, "Gene content": BLEU, "Orthologue families": ORANGE}
mark = {"Catalogue": "o", "Gene content": "s", "Orthologue families": "^"}
for nom, d in A.items():
    m = (d[d.modele == "gb"].groupby(["espece", "region"]).chute.median()
         .reset_index().merge(reg, on=["espece", "region"]))
    r, p = pearsonr(m.lignees_vues_pct, m.chute)
    b.scatter(m.lignees_vues_pct, m.chute, s=26, color=coul[nom],
              marker=mark[nom], edgecolor="white", linewidth=0.5,
              label=f"{nom}  $r={r:+.2f}$" + ("" if p >= .05 else "$^{*}$"))
    z = np.polyfit(m.lignees_vues_pct, m.chute, 1)
    xs = np.linspace(m.lignees_vues_pct.min(), m.lignees_vues_pct.max(), 20)
    b.plot(xs, np.polyval(z, xs), color=coul[nom], lw=1.0,
           ls="-" if p < .05 else ":", alpha=0.8)
b.axhline(0, color=ENCRE, lw=0.8)
b.set_xlabel("Held-out lineages already seen in training (%)")
b.set_ylabel("AUROC difference (median per region)")
b.set_title("Lineage novelty", loc="left", color=ENCRE, pad=6)
b.legend(loc="lower center", bbox_to_anchor=(0.52, -0.02), fontsize=6.8,
         handletextpad=0.2, borderpad=0.3, labelspacing=0.28,
         framealpha=0.92, frameon=True, edgecolor="none", facecolor="white")
fig.legend(*a.get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(0.5, 1.10), ncol=6, handletextpad=0.2,
           columnspacing=1.0, fontsize=7)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_confusions.{ext}")
print("ecrit fig3_confusions.pdf et .png")
