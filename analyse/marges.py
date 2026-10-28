import os
from pathlib import Path
import pandas as pd, numpy as np
P = Path(os.environ.get("CABBAGE_RESULTS", "resultats"))
OUT = Path(os.environ.get("CABBAGE_ANALYSIS_OUTPUT", "analysis_output")); OUT.mkdir(parents=True, exist_ok=True)
src = {k: P / f"{k}_resultats_forts_tous.csv" for k in ("amr", "pan", "pg")}
PROTO = ["aleatoire", "lignee", "periode", "region"]

def panneau(f):
    d = pd.read_csv(f)
    p = d.pivot_table(index=["espece", "antibiotique", "decoupage"], columns="modele",
                      values="auroc").dropna().reset_index()
    cnt = p.groupby(["espece", "antibiotique"]).decoupage.nunique()
    comm = cnt[cnt == 4].index
    p = p.set_index(["espece", "antibiotique"]).loc[comm].reset_index()
    for m in ["gb", "logistique", "rf"]:
        p["m_" + m] = p[m] - p["regles"]
    return p, comm

rng = np.random.default_rng(0)
def ic(v, B=10000):
    v = np.asarray(v); b = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return np.percentile(b, [2.5, 97.5])

A, C = {}, {}
for k, f in src.items():
    A[k], C[k] = panneau(f)
    print("%s : %d cellules, %d antibiotiques, %d especes" %
          (k, len(C[k]), A[k].antibiotique.nunique(), A[k].espece.nunique()))
comm = C["amr"].intersection(C["pan"]).intersection(C["pg"])
print("panneau commun aux trois representations : %d cellules" % len(comm))

print("\n=== TABLEAU 7 ===")
lignes = []
for k in src:
    p = A[k].set_index(["espece", "antibiotique"]).loc[comm].reset_index()
    for m in ["gb", "logistique", "rf"]:
        l = {"repr": k, "modele": m}
        for dec in PROTO:
            v = p[p.decoupage == dec]["m_" + m].dropna()
            md = np.median(v); lo, hi = ic(v)
            l[dec] = round(md, 3); l[dec + "_ic"] = "[%.3f, %.3f]" % (lo, hi)
        for dec in PROTO[1:]:
            l[dec + "_pct"] = round(100 * (l[dec] - l["aleatoire"]) / l["aleatoire"])
        lignes.append(l)
T = pd.DataFrame(lignes)
print(T[["repr", "modele", "aleatoire", "lignee", "lignee_pct", "periode", "periode_pct",
         "region", "region_pct"]].to_string(index=False))
print()
print(T[["repr", "modele"] + [d + "_ic" for d in PROTO]].to_string(index=False))
T.to_csv(OUT / "marges_tableau7.csv", index=False)

print("\n=== degradation de la marge sous holdout geographique, par espece ===")
for k in src:
    p = A[k].set_index(["espece", "antibiotique"]).loc[comm].reset_index()
    for m in ["gb", "logistique", "rf"]:
        n = 0
        for esp, s in p.groupby("espece"):
            a = s[s.decoupage == "aleatoire"]["m_" + m].median()
            r = s[s.decoupage == "region"]["m_" + m].median()
            n += (r < a)
        print("  %s %-11s : %d/6 especes ou la marge diminue" % (k, m, n))
