#!/usr/bin/env python3
"""Verify the released CABBAGE-Bench matrices, splits and headline results."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_data(root: Path) -> list[dict]:
    rows = []
    required = {"random", "lineage", "geography", "time"}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        y, m = np.load(d / "Y.npy"), np.load(d / "M.npy")
        meta = pd.read_csv(d / "meta.csv", index_col=0)
        splits = pd.read_csv(d / "splits.csv", index_col=0)
        atb = pd.read_csv(d / "antibiotiques.csv")
        mats = [sparse.load_npz(d / n) for n in ("X_genes.npz", "X_genes_pan.npz", "X_genes_pg.npz")]
        assert y.shape == m.shape == (len(meta), len(atb)), d
        assert all(x.shape[0] == len(meta) for x in mats), d
        assert set(np.unique(m)).issubset({0, 1}), d
        assert np.all(y[m == 0] == 0), f"non-zero placeholder under missing mask: {d.name}"
        protocols = set()
        for col in splits.columns:
            low = col.lower()
            for p in required:
                if p in low or {"random":"aleatoire","lineage":"lignee","geography":"region","time":"periode"}[p] in low:
                    protocols.add(p)
            vals = set(splits[col].dropna().astype(str).str.lower())
            if vals & {"train", "test"}:
                assert not splits.index.duplicated().any(), d
        assert protocols == required, f"protocols found={protocols}: {d.name}"
        summary = json.loads((d / "resume.json").read_text(encoding="utf-8"))
        assert summary["isolats"] == len(meta)
        rows.append({"species": summary["espece"], "isolates": len(meta),
                     "annotations": summary["annotations"], "drugs": len(atb),
                     "observed_pairs": int(m.sum()), "protocols": 4})
    assert sum(x["isolates"] for x in rows) == 61247
    assert sum(x["annotations"] for x in rows) == 58861
    return rows


def verify_results(root: Path) -> dict:
    amr = pd.read_csv(root / "amr_resultats_forts_tous.csv")
    loro = pd.read_csv(root / "amr_loro_tous.csv")
    regions = pd.read_csv(root / "amr_loro_regions_tous.csv")
    discord = pd.read_csv(root / "desaccord_normes.csv")
    out = {"result_rows": len(amr), "region_rows": len(regions), "discordance_rows": len(discord)}
    # Geographic comparison: one mean control per matched cell, then region-level aggregation.
    keys = ["espece", "region", "modele", "antibiotique"]
    q = loro[loro["modele"] == "gb"]
    held = q[q["type"] == "region"].groupby(keys).auroc.mean()
    ctrl = q[q["type"] != "region"].groupby(keys).auroc.mean()
    paired = pd.concat([held.rename("held"), ctrl.rename("control")], axis=1).dropna()
    paired["delta"] = paired.held - paired.control
    region_delta = paired.groupby(level=["espece", "region"]).delta.median()
    out.update({"geographic_cells": len(paired), "regions": len(region_delta),
                "regions_degraded": int((region_delta < 0).sum()),
                "median_region_delta": float(region_delta.median()),
                "region_wilcoxon_p": float(wilcoxon(region_delta).pvalue)})
    # Balanced 105-cell panel and internal rule-vs-GB margins.
    a = amr
    piv = a.pivot_table(index=["espece", "antibiotique", "modele"], columns="decoupage", values="auroc")
    common = piv.dropna().reset_index()
    drugs = common.groupby(["espece", "antibiotique"]).filter(lambda x: {"gb", "regles"}.issubset(set(x.modele)))
    margins = drugs.pivot_table(index=["espece", "antibiotique"], columns=["modele"], values=list(set(a.decoupage)))
    out["balanced_cells"] = len(margins)
    if "gb" in margins.columns.get_level_values(1) and "regles" in margins.columns.get_level_values(1):
        for protocol in ("aleatoire", "region"):
            if protocol in margins.columns.get_level_values(0):
                out[f"gb_rule_margin_{protocol}"] = float((margins[(protocol,"gb")] - margins[(protocol,"regles")]).median())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = {"data": verify_data(a.data), "results": verify_results(a.results)}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.output: a.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
