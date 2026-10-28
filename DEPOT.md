# Dépôt Zenodo prévu

Statut : préparation locale; aucun dépôt public et aucun DOI actif.

La stratégie validée est un téléversement manuel dans **un seul enregistrement
Zenodo versionné**, contenant exactement :

1. `cabbage-bench-depot.zip` -- code MIT, résultats et documentation;
2. `cabbage-bench-donnees.tgz` -- 78 fichiers dérivés sous CC BY 4.0.

L'intégration GitHub--Zenodo n'est pas utilisée pour produire cet
enregistrement, car elle archive la release GitHub et ne garantit pas que les
deux archives scientifiques demandées soient transférées comme assets.

Avant publication :

- vérifier les noms, tailles et SHA-256 des deux fichiers;
- recopier les métadonnées de `.zenodo.json`;
- indiquer distinctement la licence MIT du code et CC BY 4.0 des données;
- lier CABBAGE et son article source;
- ajouter l'URL GitHub uniquement après sa création effective;
- réserver un DOI seulement dans le brouillon destiné à être publié;
- ne jamais présenter un DOI réservé comme public.

Une fois publié, le DOI de version doit être utilisé par le manuscrit BIB.
Zenodo créera également un DOI conceptuel couvrant toutes les versions.

