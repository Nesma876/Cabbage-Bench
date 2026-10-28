# Reproducibility report

## Exécution réalisée

Environnement d'audit isolé : Python 3.12; NumPy 2.5.4; pandas 3.1.0; SciPy
1.18.1. L'installation à partir de `requirements.txt` a été testée. Ces
versions ne sont pas présentées comme l'environnement historique du calcul.

Commandes principales :

```powershell
python scripts/verify_release.py --data work/data/bench --results resultats
python analyse/tableau7.py
python analyse/decomp_apparie.py
python analyse/marges.py
python analyse/lignee_controle.py
```

Le validateur charge les 18 matrices creuses, les 12 tableaux NumPy de labels
et masques, les métadonnées, les six fichiers de partitions et les résultats.
Il contrôle les dimensions, le masque binaire, l'alignement des lignes, les
quatre protocoles, les totaux de cohorte et les agrégats principaux.

## Niveaux de preuve

- Vérifié par réexécution : chargement des matrices, masques, cohorte,
  partitions, agrégations LOORO, représentations, désaccord de normes,
  décomposition et marges de modèles.
- Vérifié dans une sortie archivée : métriques par cellule, very-major error,
  ECE, résultats mono-standard et analyses supplémentaires décrites dans les
  CSV/LaTeX.
- Partiellement vérifié : absence de fuite. Les affectations fixes sont
  uniques par isolat, mais l'audit de proximité clonale archivé montre une
  apparentement résiduel et ne prouve pas l'indépendance phylogénétique.
- Impossible à vérifier : réentraînement complet et recalcul des métriques au
  niveau individuel, faute de probabilités, modèles ajustés, fichiers
  `jeu.pkl`, données CABBAGE amont et partitions des dix redécoupages.

La compilation intégrée LaTeX a atteint six pages pour le manuscrit et dix
pages pour le supplément avant de s'arrêter sur les premières images. Le
compilateur intégré ne copie pas les fichiers auxiliaires voisins dans son
contexte isolé; aucun PDF complet n'a donc été produit. Les citations sont
également signalées comme non résolues lors de ce passage unique. Une
compilation de projet complète avec figures et BibTeX reste requise.

## Limites et commandes de reproduction complète

La reconstruction depuis CABBAGE requiert les Parquet amont, les annotations
GFF, AMRFinderPlus et MMseqs2, dont les versions exactes ne sont pas consignées.
Après récupération légitime de ces ressources, suivre `pipeline/run_all.py`,
`pipeline/baselines_forts.py`, `pipeline/geographie.py`, puis les scripts
`analyse/`. Une reproduction complète doit être effectuée dans un environnement
verrouillé et comparée aux CSV sans remplacer silencieusement ceux-ci.

