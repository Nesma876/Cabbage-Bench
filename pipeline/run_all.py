#!/usr/bin/env python3
"""
CABBAGE-Bench — chaine complete, pour execution sur serveur.

Lance tout, sans surveillance, pour plusieurs especes :
  1. matrices multi-label depuis CABBAGE
  2. filtrage des antibiotiques (tests de diagnostic, veterinaires)
  3. telechargement des annotations GFF
  4. regroupement en lignees genetiques
  5. les quatre decoupages, puis les modeles de reference
  6. ecriture des fichiers du benchmark et des resultats

REPRENABLE : chaque etape ecrit son resultat sur disque et est sautee si le
fichier existe deja. Une coupure ne fait donc jamais tout recommencer.

Usage :
    nohup python run_all.py --sortie bench/ --fils 16 > run.log 2>&1 &
    tail -f run.log

    # une seule espece, pour essayer
    python run_all.py --especes "Klebsiella pneumoniae" --limite-dl 50

    # reprendre apres une coupure : la meme commande suffit
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Le code du benchmark est celui de cellule_tout.py : on le charge tel quel
# pour qu'il n'existe qu'une seule version de la logique.
# ---------------------------------------------------------------------------
ICI = Path(__file__).resolve().parent
NOYAU = ICI / "cellule_tout.py"

ESPECES_DEFAUT = [
    "Klebsiella pneumoniae",
    "Escherichia coli",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Staphylococcus aureus",
    "Salmonella enterica",
]


def charger_noyau():
    """Execute cellule_tout.py sans ses tests de fin, et renvoie ses fonctions."""
    src = NOYAU.read_text(encoding="utf-8")
    src = src.split('print("Pret.")')[0]          # on coupe avant les tests
    espace = {}
    exec(compile(src, str(NOYAU), "exec"), espace)
    return espace


def journal(fichier):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(fichier), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("bench")


# ---------------------------------------------------------------------------
# Cache de genes compact : on stocke des entiers plutot que des chaines.
# Pour six especes, cela divise la taille du fichier par environ dix.
# ---------------------------------------------------------------------------

def compacter(cache_str):
    vocab = sorted({g for s in cache_str.values() for g in s})
    idx = {g: i for i, g in enumerate(vocab)}
    return {"vocab": vocab,
            "genomes": {a: np.array(sorted(idx[g] for g in s), dtype=np.int32)
                        for a, s in cache_str.items()}}


def decompacter(compact):
    v = compact["vocab"]
    return {a: {v[i] for i in arr} for a, arr in compact["genomes"].items()}


def charger_cache(chemin):
    """Accepte l'ancien format (ensembles de chaines) et le nouveau (compact)."""
    if not os.path.exists(chemin):
        return {}
    with open(chemin, "rb") as f:
        brut = pickle.load(f)
    if isinstance(brut, dict) and "vocab" in brut and "genomes" in brut:
        return decompacter(brut)
    return brut                                    # ancien format


def ecrire_cache(cache, chemin):
    with open(chemin, "wb") as f:
        pickle.dump(compacter(cache), f, protocol=4)


# ---------------------------------------------------------------------------

def traiter_espece(espece, args, N, log):
    slug = espece.lower().replace(" ", "_").replace(".", "")
    dossier = Path(args.sortie) / slug
    dossier.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info(f"=== {espece} ===")

    # --- 1-2. matrices puis filtrage -----------------------------------
    f_jeu = dossier / "jeu.pkl"
    if f_jeu.exists() and not args.refaire:
        log.info("  matrices : deja faites, on saute")
        with open(f_jeu, "rb") as f:
            d = pickle.load(f)
    else:
        d = N["construire"](espece, release=args.release, seuil=args.seuil)
        d = N["filtrer"](d, contexte=args.contexte)
        with open(f_jeu, "wb") as f:
            pickle.dump(d, f, protocol=4)
    n, k = d["Y"].shape
    log.info(f"  {n} isolats x {k} antibiotiques, "
             f"{100*d['M'].mean():.1f} % de cases testees")

    # --- 3. annotations -------------------------------------------------
    f_cache = dossier / "genes.pkl"
    cache = charger_cache(f_cache)
    manquants = [a for a in d["meta"]["assembly_ID"].dropna().unique()
                 if a not in cache]
    if manquants:
        log.info(f"  telechargement : {len(manquants)} annotations a recuperer")
        # on reutilise la fonction du noyau, qui gere la reprise et les erreurs
        tmp = dossier / "genes_brut.pkl"
        if cache and not tmp.exists():
            with open(tmp, "wb") as f:
                pickle.dump(cache, f, protocol=4)
        cache = N["telecharger_genes"](d["meta"], str(tmp),
                                       limite=args.limite_dl, fils=args.fils)
        ecrire_cache(cache, f_cache)
        if tmp.exists():
            tmp.unlink()
    else:
        log.info(f"  telechargement : {len(cache)} annotations deja en cache")

    # --- 4. lignees -----------------------------------------------------
    f_lig = dossier / "lignees.csv"
    if f_lig.exists() and not args.refaire:
        lignees = pd.read_csv(f_lig, index_col=0)["lignee"]
        log.info(f"  lignees : deja faites ({lignees.nunique()} groupes)")
    else:
        # on exclut les genes de resistance de CABBAGE ET ceux reperables au
        # prefixe : le portail n'annote pas toujours tout sous le meme nom
        prefixes = ("bla", "aac", "aph", "ant", "sul", "tet", "erm", "mcr",
                    "qnr", "van", "oqx", "dfr", "cat", "mph", "arm", "rmt")
        tous_genes = {g for v in cache.values() for g in v}
        amr = set(d["genes"]) | {g for g in tous_genes
                                 if g.lower().startswith(prefixes)}
        log.info(f"  {len(amr)} genes de resistance exclus du regroupement")
        lignees = N["construire_lignees"](cache, genes_amr=amr,
                                          seuil=args.seuil_jaccard,
                                          max_go=args.max_go)
        lignees.to_csv(f_lig)

    # --- 5. decoupages puis modeles -------------------------------------
    d["splits"]["lignee"] = N["splits_par_lignee"](d["meta"], lignees)
    fuite = N["controle_fuite"](d["meta"], lignees, d["splits"])
    d["splits"].to_csv(dossier / "splits.csv")

    res = N["evaluer"](d)
    res.insert(0, "espece", espece)
    res.to_csv(dossier / "resultats.csv", index=False)

    # --- 6. fichiers du benchmark ---------------------------------------
    np.save(dossier / "Y.npy", d["Y"])
    np.save(dossier / "M.npy", d["M"])
    sparse.save_npz(dossier / "X_genes.npz", d["X"])
    d["meta"].to_csv(dossier / "meta.csv")
    pd.Series(d["antibiotiques"]).to_csv(dossier / "antibiotiques.csv",
                                         index=False, header=["antibiotique"])

    resume = {
        "espece": espece,
        "isolats": int(n),
        "antibiotiques": int(k),
        "densite_pct": round(100 * float(d["M"].mean()), 1),
        "genes_amr": len(d["genes"]),
        "annotations": len(cache),
        "lignees": int(lignees.nunique()),
        "fuite_lignees": len(fuite),
        "antibiotiques_exclus": d.get("exclus", {}),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(dossier / "resume.json", "w") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    log.info(f"  termine en {resume['minutes']} min — {resume['lignees']} lignees")
    return resume, res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--especes", nargs="*", default=ESPECES_DEFAUT)
    p.add_argument("--sortie", default="bench")
    p.add_argument("--release", default="2025-11")
    p.add_argument("--seuil", type=int, default=300,
                   help="isolats testes minimum par antibiotique")
    p.add_argument("--seuil-jaccard", type=float, default=None,
                   help="seuil de regroupement ; par defaut CALIBRE par espece. "
                        "Ne le fixer qu'en connaissance de cause : un seuil "
                        "trop strict isole les genomes et rend le decoupage "
                        "par lignee equivalent a un tirage aleatoire.")
    p.add_argument("--max-go", type=float, default=8,
                   help="memoire au-dela de laquelle on regroupe sur echantillon")
    p.add_argument("--fils", type=int, default=16,
                   help="telechargements en parallele")
    p.add_argument("--limite-dl", type=int, default=None,
                   help="plafond d'annotations, pour un essai rapide")
    p.add_argument("--contexte", default=None, choices=[None, "humain", "animal"])
    p.add_argument("--refaire", action="store_true",
                   help="recalculer meme si les fichiers existent")
    args = p.parse_args()

    Path(args.sortie).mkdir(parents=True, exist_ok=True)
    log = journal(Path(args.sortie) / "run.log")
    log.info(f"CABBAGE-Bench — {len(args.especes)} especes, release {args.release}")

    N = charger_noyau()
    resumes, tous = [], []
    for espece in args.especes:
        try:
            r, res = traiter_espece(espece, args, N, log)
            resumes.append(r)
            tous.append(res)
        except Exception as e:                                  # noqa: BLE001
            log.exception(f"  ECHEC sur {espece} : {e}")
            log.info("  on passe a l'espece suivante ; relancer reprendra ici")

    if resumes:
        df = pd.DataFrame(resumes).drop(columns=["antibiotiques_exclus"])
        df.to_csv(Path(args.sortie) / "resume_especes.csv", index=False)
        log.info("\n=== Resume ===\n" + df.to_string(index=False))

    if tous:
        gros = pd.concat(tous, ignore_index=True)
        gros.to_csv(Path(args.sortie) / "resultats_tous.csv", index=False)
        log.info("\n=== AUROC moyen par decoupage et modele ===\n" +
                 gros.pivot_table(index=["espece", "decoupage"], columns="modele",
                                  values="auroc", aggfunc="mean")
                     .round(3).to_string())
        log.info("\n=== Erreur tres majeure moyenne (%) ===\n" +
                 gros.pivot_table(index=["espece", "decoupage"], columns="modele",
                                  values="err_tres_majeure_pct", aggfunc="mean")
                     .round(1).to_string())
    log.info("Fini.")


if __name__ == "__main__":
    main()
