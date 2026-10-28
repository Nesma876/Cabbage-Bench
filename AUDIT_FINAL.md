# CABBAGE-Bench -- audit scientifique et technique

Date de l'audit : 2026-10-28. Statut : **prépublication bloquée**.

## Conclusion

Les archives sont lisibles et l'empreinte annoncée pour les données est
exacte. Les 78 fichiers de données décrivent bien six espèces, 61 247 isolats,
58 861 annotations et quatre affectations de partition. Les résultats
principaux recalculables depuis les CSV concordent avec le manuscrit : 424
paires région--antibiotique descriptives, 20/21 régions en dégradation,
médiane -0.09324 et Wilcoxon régional p=2.384e-5; 586 506 paires
CLSI/EUCAST dont 48 368 discordantes (8.247%); décomposition croisée sur 22
unités, p=0.4245; panneau de 105 cellules, marges 0.1248 et 0.0192.

La provenance et la redistribution sous CC BY 4.0 ont depuis été confirmées
par les auteurs et contrôlées contre l'article CABBAGE et le portail EMBL-EBI.
La publication publique n'est toutefois pas autorisée à ce stade : le DOI
n'existe pas et la conformité du flux rédactionnel assisté par ChatGPT doit
être clarifiée avec BIB; versions et dates des outils restent à compléter.

## Intégrité

| Archive reçue | Octets | Fichiers extraits | SHA-256 | Statut |
|---|---:|---:|---|---|
| `cabbage-bench-depot (1).zip` | 1 380 221 | 37 fichiers (42 entrées avec répertoires) | `2d56339a252ce76fe7324cd786a63f6e1f3baf13260b2799683bde4668f411f0` | ZIP lisible; empreinte absente de `pour_nesma.md` |
| `cabbage-bench-donnees.tgz` | 73 396 037 | 78 | `57e2f5f70326a044c266852ed546672936b0d37ae6b9e83fb4f50ecfb9348171` | conforme à `pour_nesma.md` |

Aucun fichier extrait n'est vide. Le nom du ZIP reçu diffère du nom de dépôt
prévu. Un renommage sans modification des octets est possible, mais toute
archive corrigée du code aura nécessairement une nouvelle empreinte et devra
être distinguée de l'originale.

## Résultats vérifiés

| Résultat | Valeur manuscrit | Valeur recalculée | Méthode | Statut |
|---|---:|---:|---|---|
| Isolats | 61 247 | 61 247 | Somme des six `resume.json` et dimensions de `Y` | vérifié par exécution |
| Génomes annotés | 58 861 | 58 861 | Somme des six résumés | vérifié par exécution |
| LOORO | 424 paires; 20/21; -0.093; p=2.4e-5 | 424; 20/21; -0.09324; 2.384e-5 | appariement GB, médiane par région | vérifié par exécution |
| Représentations | 411; -0.101/-0.132/-0.142 | sortie appariée: -0.10104/-0.13201/-0.14184 | 21 agrégats régionaux | vérifié par exécution |
| CLSI/EUCAST | 586 506; 8.25% | 586 506; 48 368 = 8.247% | somme `desaccord_normes.csv` | vérifié par exécution |
| Décomposition | 22; p=0.42 | 22; p=0.42449 | modèles moyennés avant Wilcoxon | vérifié par exécution |
| Heuristique vs GB | 0.125; 0.019 | 0.1248; 0.0192 | panneau complet de 105 cellules | vérifié par exécution |

Les valeurs de calibration et de very-major error sont présentes dans les
sorties archivées. L'absence de probabilités individuelles empêche de les
recalculer à partir des prédictions; leur statut est donc « vérifié directement
dans une sortie archivée », pas « réexécuté ».

## Anomalies et corrections

1. Cinq scripts d'analyse contenaient des chemins absolus d'une ancienne
   machine. Les scripts utilisés pour les principaux tableaux ont été rendus
   relatifs à `resultats/`.
2. `pipeline/baselines_forts.py` exigeait `jeu.pkl`, absent des données. Il
   charge désormais le vocabulaire AMR depuis le bloc AMR de `genes_pan.csv`.
3. Le script de décomposition testait 44 observations dépendantes et donnait
   p=0.2296. Il moyenne maintenant les familles de modèles pour obtenir les 22
   unités primaires et p=0.4245.
4. `.zenodo.json` décrit à tort les données comme déposées séparément et
   « plusieurs gigaoctets »; elles font 73 396 037 octets compressés et doivent
   être dans le même enregistrement. Ce fichier doit être harmonisé avant dépôt.
5. Les affiliations diffèrent entre `pour_nesma.md`, `.zenodo.json` et le
   manuscrit; la graphie `Yncréa/Yncrea` et `LabISEN/L@bISEN` doit être validée.
6. L'archive ne verrouille pas les versions exactes de l'environnement de
   calcul original et ne contient ni modèles ajustés ni probabilités par isolat.
7. Les 424 comparaisons descriptives deviennent 411 cellules dans certaines
   analyses filtrées; cette différence doit être explicitée uniformément.

## Diffusion et conformité BIB

BIB exige une déclaration de disponibilité et un dépôt émettant un DOI. Les
jeux publics doivent aussi être cités dans la bibliographie avec le marqueur
`[dataset]`. La déclaration d'IA actuelle est insuffisante : BIB demande une
déclaration dans la lettre et le manuscrit, avec documentation détaillée dans
le supplément; la politique OUP mise à jour le 13 juillet 2026 demande le nom
et la version de l'outil, son usage et la supervision humaine. BIB interdit en
outre l'emploi d'un LLM pour rédiger un article à partir d'un prompt. La phrase
actuelle mentionnant « manuscript drafting » doit donc être examinée par les
auteurs et, si nécessaire, clarifiée auprès de la rédaction avant soumission.

## Décision

Ne pas publier GitHub/Zenodo et ne pas insérer de DOI avant résolution des
métadonnées restantes et de la déclaration d'IA. Aucun résultat scientifique
n'a été modifié pour obtenir une concordance.

