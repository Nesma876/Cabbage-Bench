import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
P = Path(os.environ.get("CABBAGE_RESULTS", "resultats"))
OUT = Path(os.environ.get("CABBAGE_ANALYSIS_OUTPUT", "analysis_output")); OUT.mkdir(parents=True, exist_ok=True)
src = {k: P for k in ("amr", "pan", "pg")}
f = {k: f"{k}_resultats_forts_tous.csv" for k in src}
l = {k: f"{k}_loro_tous.csv" for k in src}

F = pd.concat([pd.read_csv(src[k] / f[k]).assign(repr=k) for k in src])
L = pd.concat([pd.read_csv(src[k] / l[k]).assign(repr=k) for k in src])
print("especes par representation (forts) :")
for k, s in F.groupby("repr"): print(" ", k, sorted(s.espece.unique()), len(s))
print("\nespeces par representation (loro) :")
for k, s in L.groupby("repr"): print(" ", k, sorted(s.espece.unique()), len(s))

ESP_PG = sorted(F[F["repr"] == "pg"].espece.unique())
print("\n=== TABLEAU 7 : AUROC moyen par protocole et representation ===")
print("--- (a) toutes les especes disponibles pour chaque representation")
t = F[F.modele != "regles"].pivot_table(index=["repr", "modele"], columns="decoupage",
                                        values="auroc", aggfunc="mean").round(3)
print(t.to_string())
print("\n--- (b) restreint aux especes du pangenome :", ESP_PG)
Fr = F[F.espece.isin(ESP_PG)]
t2 = Fr.pivot_table(index=["repr", "modele"], columns="decoupage", values="auroc",
                    aggfunc="mean").round(3)
print(t2.to_string())
t2.to_csv(OUT / "tableau7.csv")

print("\n=== comparaison appariee par (espece, antibiotique, decoupage, modele) ===")
cle = ["espece", "antibiotique", "decoupage", "modele"]
w = Fr[Fr.modele != "regles"].pivot_table(index=cle, columns="repr", values="auroc").dropna()
print("cellules appariees :", len(w))
for a, b in [("amr", "pan"), ("amr", "pg"), ("pan", "pg")]:
    st = wilcoxon(w[a], w[b])
    print("%s %.4f  vs  %s %.4f   delta=%+.4f  p=%.3g" %
          (a, w[a].mean(), b, w[b].mean(), w[b].mean() - w[a].mean(), st.pvalue))

print("\npar protocole (gb seul, appariement complet) :")
w2 = Fr[Fr.modele == "gb"].pivot_table(index=cle, columns="repr", values="auroc").dropna().reset_index()
for dec, s in w2.groupby("decoupage"):
    print("  %-10s n=%4d  amr %.3f  pan %.3f  pg %.3f" %
          (dec, len(s), s.amr.mean(), s.pan.mean(), s.pg.mean()))

print("\n=== LORO : region tenue vs temoins apparies, par representation ===")
cleL = ["espece", "region", "modele", "antibiotique"]
rows = []
for k, s in L.groupby("repr"):
    s = s[s.n_test >= 30]
    ten = s[s.type == "region"].groupby(cleL).auroc.mean().rename("tenu")
    tem = s[s.type != "region"].groupby(cleL).auroc.mean().rename("temoin")
    m = pd.concat([ten, tem], axis=1).dropna()
    m["delta"] = m.tenu - m.temoin
    g = m.groupby(level=["espece", "region"]).delta.mean()
    rng = np.random.default_rng(0); v = g.to_numpy()
    bs = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
    rows.append(dict(repr=k, n_cellules=len(m), n_grappes=len(v),
                     delta=round(v.mean(), 4), ic_bas=round(np.percentile(bs, 2.5), 4),
                     ic_haut=round(np.percentile(bs, 97.5), 4),
                     degradent_pct=round(100 * (m.delta < 0).mean(), 1),
                     p=wilcoxon(m.tenu, m.temoin).pvalue))
print(pd.DataFrame(rows).to_string(index=False))
print("\ntypes dans loro :", L.type.unique())
