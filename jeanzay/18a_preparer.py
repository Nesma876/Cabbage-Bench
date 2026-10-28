#!/usr/bin/env python3
"""Etape A : deposer les parquet sur le disque et recopier le cache GFF.

A lancer SUR LA FRONTALE. On passe par DuckDB et non par curl : c'est le
mecanisme qui lit deja ces fichiers dans toute la chaine, donc s'il marche
pour lire il marche pour copier, alors que curl se fait renvoyer un 403.
"""
import os, shutil, sys
from pathlib import Path
import duckdb

BASE = Path("/lustre/fswork/projects/rech/rme/ufb85tk/pfe_tlm")
REL = "2025-11"
FTP = f"https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases/{REL}"
PQ = BASE / "parquet" / REL
PQ.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
try:
    con.execute("INSTALL httpfs; LOAD httpfs;")
except Exception as e:
    print(f"httpfs : {e} -- on continue, l'extension est peut-etre deja la")
con.execute("SET memory_limit='8GB'; SET threads=4;")
for quoi in ("phenotype", "genotype"):
    out = PQ / f"{quoi}.parquet"
    if out.exists() and out.stat().st_size > 0:
        print(f"deja la : {quoi}.parquet ({out.stat().st_size/1e6:.0f} Mo)")
        continue
    print(f"copie de {quoi}.parquet ...", flush=True)
    tmp = out.with_suffix(".parquet.tmp")
    con.execute(f"COPY (SELECT * FROM read_parquet('{FTP}/{quoi}.parquet')) "
                f"TO '{tmp}' (FORMAT PARQUET)")
    tmp.rename(out)
    print(f"  ecrit : {out.stat().st_size/1e6:.0f} Mo")

n = 0
for norme in ("clsi", "eucast"):
    for d in sorted((BASE / "bench").glob("*/")):
        src = d / "genes.pkl"
        if not src.exists():
            continue
        dst = BASE / f"bench_norme_{norme}" / d.name / "genes.pkl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst); n += 1
print(f"cache d'annotations : {n} fichiers recopies")
print("\nPret. Lance maintenant :  sbatch 18b_normes_calcul.slurm")
