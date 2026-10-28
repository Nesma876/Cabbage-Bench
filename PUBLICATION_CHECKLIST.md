# Publication checklist

## Bloquants avant publication

- [x] Obtenir une confirmation écrite des droits de redistribution des matrices dérivées et des métadonnées CABBAGE/EBI.
- [x] Harmoniser les deux noms d'auteur et les affiliations avec le manuscrit.
- [ ] Confirmer les modèles/versions et dates d'utilisation de Claude et ChatGPT, puis harmoniser manuscrit, supplément et lettre d'accompagnement.
- [ ] Remplacer les bornes de dépendances par un environnement verrouillé testé sur une machine propre.
- [x] Reconstituer `cabbage-bench-depot.zip`, conserver l'archive originale et publier l'empreinte de chaque version.
- [ ] Créer le brouillon Zenodo unique et téléverser exactement les deux archives prévues.
- [ ] Vérifier noms, tailles, empreintes, licence, métadonnées et DOI réservé.
- [ ] Demander l'autorisation explicite de publication.

## Manuscrit BIB

- [ ] Ajouter le DOI public vérifié et une citation de jeu de données `[dataset]` dans la bibliographie.
- [ ] Ajouter l'URL GitHub et le tag uniquement s'ils existent réellement.
- [x] Employer 22 unités primaires et `paired p = 0.42`.
- [x] Décrire l'heuristique comme comparateur interne, pas comme ResFinder/AMRFinderPlus.
- [ ] Résoudre l'incohérence 424 cellules descriptives vs 411 cellules après seuil explicite dans le supplément.
- [ ] Compiler et inspecter visuellement le manuscrit et le supplément (paquets Overleaf autonomes préparés; aucun moteur LaTeX de projet disponible localement).

## Après publication

- [ ] Résoudre le DOI depuis un navigateur indépendant.
- [ ] Télécharger les deux fichiers et recalculer SHA-256.
- [ ] Vérifier le commit, la branche, le tag et la release GitHub.
- [ ] Mettre à jour `DATA_AVAILABILITY_FINAL.tex` et `DEPOSIT_REPORT.md`.

