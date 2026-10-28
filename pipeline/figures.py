#!/usr/bin/env python3
"""
Les figures de l'article 1, en PDF vectoriel pour LaTeX.

CHOIX DE CONCEPTION, et pourquoi
  Palette Okabe-Ito, dans l'ordre bleu / vert / orange / rose. Elle est
  validee : separation suffisante pour les trois formes de daltonisme, et
  lisible en niveaux de gris a l'impression. L'ordre des teintes n'est pas
  decoratif, il est choisi pour que les paires ADJACENTES soient les plus
  distinguables.

  Pas de couleur seule pour porter l'information : chaque serie est aussi
  identifiee par un marqueur de forme differente et par une legende. Un
  lecteur daltonien, une photocopie en noir et blanc, ou un relecteur qui
  imprime l'article doivent tous pouvoir lire la figure.

  Un seul axe par figure, jamais deux echelles. Grille discrete, en arriere.
  Marques fines. Pas de valeur imprimee sur chaque point : seulement sur ce
  qui doit etre lu.

Usage :
    python figures.py --bench $BASE/bench --sortie figures/
Les fichiers manquants sont sautes avec un avertissement : on peut donc
lancer le script avant que tout soit calcule.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- palette validee -------------------------------------------------------
BLEU, VERT, ORANGE, ROSE = "#0072B2", "#009E73", "#E69F00", "#CC79A7"
COUL = {"aleatoire": "#444444", "lignee": BLEU, "region": VERT, "periode": ORANGE}
FORME = {"aleatoire": "o", "lignee": "s", "region": "^", "periode": "D"}
NOM = {"aleatoire": "Random", "lignee": "Lineage-aware",
       "region": "Geography-aware", "periode": "Time-aware"}
MOD = {"regles": "Rule-based", "logistique": "Logistic regression",
       "rf": "Random forest",
       "gb": "Gradient boosting", "separe": "Independent per-drug",
       "multi_label": "Multi-label"}
ENCRE, ENCRE2, GRILLE = "#1a1a1a", "#595959", "#d9d9d9"


def style():
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "axes.edgecolor": ENCRE2, "axes.linewidth": 0.6,
        "axes.labelcolor": ENCRE, "axes.titlesize": 9.5,
        "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.5,
        "xtick.color": ENCRE2, "ytick.color": ENCRE2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def joli(nom):
    """klebsiella_pneumoniae -> K. pneumoniae"""
    p = str(nom).replace("_", " ").split()
    return f"{p[0][0].upper()}. {' '.join(p[1:])}" if len(p) > 1 else str(nom)


def sauver(fig, sortie, nom):
    for ext in ("pdf", "png"):
        fig.savefig(Path(sortie) / f"{nom}.{ext}")
    plt.close(fig)
    print(f"  ecrit {nom}.pdf et {nom}.png")


# ---------------------------------------------------------------------------
# Figure 1 : AUROC par espece et par decoupage
# ---------------------------------------------------------------------------

def fig1(res, sortie, modele="separe", nom="fig1_auroc_especes_decoupages"):
    """Un graphe a points, et non des barres groupees : avec 6 especes x 4
    decoupages, des barres donnent 24 rectangles illisibles, alors que des
    points alignes sur une ligne par espece se comparent d'un coup d'oeil."""
    d = res[res["modele"] == modele] if modele in set(res["modele"]) else res
    t = d.pivot_table(index="espece", columns="decoupage", values="auroc",
                      aggfunc="mean")
    cols = [c for c in ("aleatoire", "lignee", "region", "periode")
            if c in t.columns]
    t = t[cols].sort_values("aleatoire")

    fig, ax = plt.subplots(figsize=(5.4, 0.52 * len(t) + 1.1))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        vals = [r[c] for c in cols if pd.notna(r[c])]
        if len(vals) > 1:                       # le trait relie, il ne decore pas
            ax.plot([min(vals), max(vals)], [i, i], color=GRILLE, lw=1.4,
                    zorder=1, solid_capstyle="round")
    for c in cols:
        ax.scatter(t[c], y, s=42, color=COUL[c], marker=FORME[c],
                   edgecolor="white", linewidth=0.7, zorder=3, label=NOM[c])
    ax.set_yticks(y)
    ax.set_yticklabels([joli(i) for i in t.index], style="italic")
    ax.set_xlabel("Mean AUROC over antimicrobials")
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
              handletextpad=0.3, columnspacing=1.2)
    sauver(fig, sortie, nom)


# ---------------------------------------------------------------------------
# Figure 2 : distribution des chutes en leave-one-region-out
# ---------------------------------------------------------------------------

def fig2(loro, sortie):
    cle = ["espece", "region", "modele", "antibiotique"]
    g = loro.pivot_table(index=cle, columns="type", values="auroc")
    g = g.rename(columns={"region": "a_region", "temoin": "a_temoin"}).dropna()
    if g.empty:
        print("  fig2 : aucune paire region/temoin"); return
    g["chute"] = g["a_region"] - g["a_temoin"]
    g = g.reset_index()

    especes = (g.groupby("espece")["chute"].median().sort_values().index.tolist())
    mods = [m for m in ("logistique", "rf", "gb") if m in set(g["modele"])]
    teintes = [BLEU, ORANGE, VERT][:len(mods)]
    fig, ax = plt.subplots(figsize=(5.6, 0.75 * len(especes) + 1.2))
    rng = np.random.default_rng(0)

    for i, esp in enumerate(especes):
        for k, mod in enumerate(mods):
            s = g[(g.espece == esp) & (g.modele == mod)]["chute"]
            if s.empty:
                continue
            dy = (k - (len(mods) - 1) / 2) * 0.24
            # points semes, pour montrer la dispersion reelle et non une boite
            ax.scatter(s, np.full(len(s), i + dy) + rng.normal(0, 0.045, len(s)),
                       s=9, color=teintes[k], alpha=0.42, linewidth=0,
                       zorder=2)
            med = s.median()
            ax.plot([med, med], [i + dy - 0.11, i + dy + 0.11],
                    color=teintes[k], lw=2.1, zorder=4,
                    solid_capstyle="butt")
            # La valeur se place a GAUCHE du trait, sur la ligne du modele :
            # au-dessus ou au-dessous, trois modeles font entrer les
            # etiquettes en collision des que deux medianes sont proches.
            ax.annotate(f"{med:+.3f}", (med, i + dy), xytext=(-7, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=6.5, color=teintes[k])
    ax.axvline(0, color=ENCRE, lw=0.9, zorder=1)
    ax.set_yticks(range(len(especes)))
    ax.set_yticklabels([joli(e) for e in especes], style="italic")
    ax.set_xlabel("AUROC difference, held-out region minus size-matched random control")
    ax.set_ylim(-0.7, len(especes) - 0.3)
    ax.grid(axis="y", visible=False)
    # Bornes lisibles : quelques points extremes ecrasaient toute la
    # distribution. On les compte explicitement plutot que de les cacher.
    bas, haut = -0.45, 0.25
    hors = int(((g["chute"] < bas) | (g["chute"] > haut)).sum())
    ax.set_xlim(bas, haut)
    if hors:
        ax.annotate(f"{hors} of {len(g)} points beyond axis limits",
                    (0.015, 0.015), xycoords="axes fraction", fontsize=6.5,
                    color=ENCRE2)
    ax.legend(handles=[Line2D([], [], color=teintes[k], lw=2.1,
                              label=MOD.get(m, m)) for k, m in enumerate(mods)],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3)
    sauver(fig, sortie, "fig2_leave_one_region_out")


# ---------------------------------------------------------------------------
# Figure 3 : les deux facteurs de confusion ecartes
# ---------------------------------------------------------------------------

def fig3(loro, regions, sortie):
    cle = ["espece", "region", "modele", "antibiotique"]
    g = loro.pivot_table(index=cle, columns="type", values="auroc")
    g = g.rename(columns={"region": "a_region", "temoin": "a_temoin"}).dropna()
    if g.empty:
        print("  fig3 : rien a tracer"); return
    g["chute"] = g["a_region"] - g["a_temoin"]
    p = loro[loro["type"] == "region"].set_index(cle)
    j = g.join(p[["prev_train_pct", "prev_test_pct"]], how="inner").reset_index()
    j["d_prev"] = (j["prev_test_pct"] - j["prev_train_pct"]).abs()
    j = j[j["modele"] == ("gb" if "gb" in set(j["modele"]) else j["modele"].iloc[0])]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    # les deux panneaux portent chacun leur etiquette d'axe :
    # sans ecart suffisant, celle de droite recouvre le trace de gauche
    fig.subplots_adjust(wspace=0.42)
    especes = sorted(j["espece"].unique())
    pal = [BLEU, VERT, ORANGE, ROSE, "#56B4E9", "#D55E00"]
    formes = ["o", "s", "^", "D", "v", "P"]

    # panneau gauche : la prevalence, au niveau antibiotique
    a = axes[0]
    for k, esp in enumerate(especes):
        s = j[j.espece == esp]
        a.scatter(s["d_prev"], s["chute"], s=13, color=pal[k % 6],
                  marker=formes[k % 6], alpha=0.6, linewidth=0,
                  label=joli(esp))
    a.axhline(0, color=ENCRE, lw=0.8)
    if len(j) > 10:
        r = j["chute"].corr(j["d_prev"])
        a.annotate(f"$r = {r:+.3f}$", (0.97, 0.05), xycoords="axes fraction",
                   ha="right", fontsize=8, color=ENCRE)
    a.set_xlabel("|prevalence shift| (percentage points)")
    a.set_ylabel("AUROC difference (per antimicrobial)")
    a.set_title("Not a prevalence shift", loc="left", color=ENCRE)

    # panneau droit : la lignee, au niveau region
    b = axes[1]
    if regions is not None and not regions.empty:
        m = (g.reset_index().groupby(["espece", "region"])["chute"].median()
              .reset_index().merge(regions, on=["espece", "region"], how="inner"))
        m = m.dropna(subset=["lignees_vues_pct"])
        for k, esp in enumerate(sorted(m["espece"].unique())):
            s = m[m.espece == esp]
            b.scatter(s["lignees_vues_pct"], s["chute"], s=34,
                      color=pal[k % 6], marker=formes[k % 6],
                      edgecolor="white", linewidth=0.6, label=joli(esp))
        b.axhline(0, color=ENCRE, lw=0.8)
        if len(m) > 4:
            r = m["chute"].corr(m["lignees_vues_pct"])
            b.annotate(f"$r = {r:+.3f}$", (0.97, 0.05), xycoords="axes fraction",
                       ha="right", fontsize=8, color=ENCRE)
    b.set_xlabel("Held-out lineages already seen in training (%)")
    # Les deux panneaux montrent la meme grandeur a des granularites
    # differentes — par antibiotique a gauche, mediane par region a droite —
    # donc des echelles differentes. Sans etiquette propre, le lecteur
    # croirait a un axe partage et comparerait des amplitudes incomparables.
    b.set_ylabel("AUROC difference (median per region)")
    b.set_title("Not a lineage effect", loc="left", color=ENCRE)
    b.legend(loc="lower left", bbox_to_anchor=(-0.08, 1.14), ncol=3,
             handletextpad=0.2, columnspacing=0.9, fontsize=7)
    sauver(fig, sortie, "fig3_confusions")


# ---------------------------------------------------------------------------
# Figure 4 : erreurs tres majeures, la metrique clinique
# ---------------------------------------------------------------------------

def fig4(res, sortie, modele="separe", nom="fig4_erreurs_tres_majeures"):
    if "err_tres_majeure_pct" not in res.columns:
        print("  fig4 : colonne absente"); return
    d = res[res["modele"] == modele] if modele in set(res["modele"]) else res
    t = d.pivot_table(index="espece", columns="decoupage",
                      values="err_tres_majeure_pct", aggfunc="mean")
    cols = [c for c in ("aleatoire", "lignee", "region", "periode")
            if c in t.columns]
    t = t[cols].sort_values("aleatoire")

    fig, ax = plt.subplots(figsize=(5.4, 0.52 * len(t) + 1.1))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        vals = [r[c] for c in cols if pd.notna(r[c])]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [i, i], color=GRILLE, lw=1.4, zorder=1)
    for c in cols:
        ax.scatter(t[c], y, s=42, color=COUL[c], marker=FORME[c],
                   edgecolor="white", linewidth=0.7, zorder=3, label=NOM[c])
    ax.set_yticks(y)
    ax.set_yticklabels([joli(i) for i in t.index], style="italic")
    ax.set_xlabel("Very major errors (%): resistant isolates called susceptible")
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
              handletextpad=0.3, columnspacing=1.2)
    sauver(fig, sortie, nom)


# ---------------------------------------------------------------------------

def lire(chemin, quoi):
    if not Path(chemin).exists():
        warnings.warn(f"{quoi} absent : {chemin}")
        return None
    return pd.read_csv(chemin)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--sortie", default="figures")
    a = p.parse_args()
    style()
    r = Path(a.bench)
    Path(a.sortie).mkdir(parents=True, exist_ok=True)

    res = lire(r / "resultats_tous.csv", "resultats_tous.csv")
    forts = lire(r / "resultats_forts_tous.csv", "resultats_forts_tous.csv")
    loro = lire(r / "loro_tous.csv", "loro_tous.csv")
    regions = lire(r / "loro_regions_tous.csv", "loro_regions_tous.csv")

    print("Figures :")
    if forts is not None:
        # L'annexe compare DEUX MODELES sur UNE SEULE matrice : on la tire du
        # meme fichier que les figures principales, sinon on comparerait deux
        # representations en croyant comparer deux modeles.
        fig1(forts, a.sortie, "logistique", "fig1_auroc_especes_decoupages")
        fig4(forts, a.sortie, "logistique", "fig4_erreurs_tres_majeures")
        # Les memes figures pour le gradient boosting : ce sont celles-la qui
        # iront dans l'article, puisque c'est le modele fort qui porte
        # l'argument contre l'objection "vos modeles sont trop faibles".
        fig1(forts, a.sortie, "gb", "fig1b_auroc_gradient_boosting")
        fig4(forts, a.sortie, "gb", "fig4b_erreurs_gradient_boosting")
    if loro is not None:
        fig2(loro, a.sortie)
        fig3(loro, regions, a.sortie)
    print(f"\nTout est dans {a.sortie}/ — les .pdf pour LaTeX, "
          f"les .png pour regarder.")


if __name__ == "__main__":
    main()
