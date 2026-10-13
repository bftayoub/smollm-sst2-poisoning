# Vérification des papiers et corrections

Revue du 12 octobre 2026, avec mesures GPU sur les checkpoints du pilote
existant. Aucun résultat d'efficacité supplémentaire n'est supposé.

## Interprétation de 0,721

Ce cosinus correspond à un angle d'environ 43,8°. Il reste améliorable,
mais aucun seuil validé dans notre expérience ne permet de le déclarer
suffisant. Pour une petite mise à jour SGD, la variation de la loss cible
est approximée par `−η <g_cible, g_entraînement>`. Le produit scalaire dépend
des normes, du cosinus, du mélange avec les données propres et du checkpoint.
Cette approximation ne reproduit pas les mises à jour cumulées d'AdamW.

## Sources primaires vérifiées

**Witches' Brew** permet une référence propre fixe et utilise le cosinus du
gradient moyen des poisons, Adam signé, plusieurs départs, augmentations
différentiables et, dans certaines expériences, ensembles de modèles.
Les poisons appartiennent à la classe visée ; les expériences principales
utilisent huit départs et 250 étapes. La proposition théorique suppose un
alignement persistant pendant l'entraînement, avec des conditions sur les pas.
Un cosinus final isolé n'est donc pas une garantie.
[Sections 3.3–5 et algorithme 1](https://arxiv.org/pdf/2009.02276).

**Winter Soldier** vise des secrets aléatoires hors distribution, avec une
loss autorégressive sur la séquence poison et une relaxation Gumbel-Softmax.
L'annexe B.2 donne 200 étapes, Adam signé, taux de 0,9 et température de 0,6.
Les poisons sont répétés dans le préentraînement. Leurs tailles varient selon
les sections ; l'indication d'initialisation à −15 ne spécifie pas tous les
logits. Ces détails ne constituent pas une recette identique pour SST-2.
[Sections 3.3–4.1 et annexe B.2, arXiv v1](https://arxiv.org/html/2506.14913v1).

**Data Taggants**, référence directe de Winter Soldier, utilise aussi des
clés hors distribution, des poisons associés à l'étiquette visée et plusieurs
départs sélectionnés selon l'objectif de fabrication.
[Section 3.2](https://arxiv.org/html/2410.09101).

Le code des auteurs confirme que `signAdam` applique le signe au gradient
avant Adam, avec une décroissance du taux d'apprentissage.
[Implémentation Witches' Brew](https://github.com/JonasGeiping/poisoning-gradient-matching/blob/master/forest/witchcoven/witch_base.py).
Aucun code externe n'a été copié.

**Correction du diagnostic précédent :** une référence déjà entraînée est
compatible avec ces méthodes. Il faut vérifier le transfert réel ; ce choix
ne constitue pas à lui seul une erreur de protocole. Renverser un sentiment
explicite et implanter un secret aléatoire sont également des objectifs distincts.

## Nouvelles mesures sur les 50 ajouts exacts

| Checkpoint | Cosinus aléatoire / loss cible | Cosinus optimisé / loss cible | Norme aléatoire | Norme optimisée |
|---|---:|---:|---:|---:|
| Initial | 0,586444 | 0,559932 | 38,21098 | 38,75095 |
| Référence | 0,414669 | 0,721375 | 49,32407 | 27,27228 |
| Victime propre finale | 0,442577 | 0,723567 | 49,72403 | 28,18600 |

L'avantage de l'optimisé se retrouve sur la victime propre finale, mais pas
au checkpoint initial : le transfert dépend de l'état du modèle. Le pilote
n'avait pas enregistré de checkpoints intermédiaires par époque.

Sur la référence, la projection du gradient des ajouts sur la direction cible
normalisée vaut **19,67354 pour l'optimisé**, contre **20,45314 pour l'aléatoire**.
Le gain de cosinus est compensé par une norme inférieure. Cette contribution
locale, avant pondération par le budget, ne prédit pas la réussite finale.

Le diagnostic mesure aussi le gradient de la marge
`logit_classe_réelle − logit_classe_visée`, dont la réduction correspond au
basculement recherché :

| Checkpoint | Cosinus aléatoire / marge | Cosinus optimisé / marge |
|---|---:|---:|
| Initial | −0,071300 | **−0,081010** |
| Référence | 0,414669 | 0,721375 |
| Victime propre finale | 0,442577 | 0,723568 |

Au checkpoint initial, les poisons optimisés sont positivement alignés avec
la loss cible sur le vocabulaire, mais négativement alignés avec la réduction
de la marge. Sous l'approximation SGD locale, leurs seuls gradients augmenteraient
donc cette marge. La loss sur le vocabulaire contient aussi l'apprentissage
des codes d'étiquette, indépendamment de leur ordre relatif. Ce mécanisme
constitue une interprétation locale ; il ne prouve pas une causalité sur toute
la trajectoire d'AdamW.

Les gradients combinés utilisent un échantillon propre uniforme fixé de
256 lignes et les poids réels `5000/5050` et `50/5050`. Ils sont des estimations
du gradient du corpus augmenté. Les gradients des ajouts dans les tableaux
sont calculés sur les 50 lignes complètes. Le JSON conserve les identifiants
échantillonnés et les empreintes des poids et données.

## Variante implémentée

Les configurations existantes gardent leurs paramètres par défaut.
`configs/pilot-robust.yaml` conserve la cible `validation:652`, les 5 000
exemples propres, les 50 ajouts équilibrés et la graine victime 101.
Elle modifie la fabrication des poisons :

1. Gradient cible de **cross-entropie binaire** sur les deux codes de décision.
   La loss des poisons et du fine-tuning reste sur le vocabulaire complet.
   C'est notre ablation pour viser la décision, pas une reproduction du papier.
2. **Ensemble initial + référence**, moyenne des deux objectifs avec le même
   bruit Gumbel. C'est notre adaptation pour tester le transfert observé.
3. **Adam signé**, taux initial de 0,9, décroissance en trois paliers et
   température fixée à 0,6.
4. Tokens initiaux aléatoires : logit 0 pour le token choisi, −15 pour les
   alternatives. Cette interprétation est explicite ; mettre tous les logits
   à −15 laisserait la distribution uniforme.
5. **Trois départs** ; tous les 20 pas, évaluation d'un argmax et d'un
   échantillon catégoriel, après décodage et retokenisation. Sélection selon
   le cosinus discret moyen des deux checkpoints propres, sans résultats victimes.

Les scores par checkpoint, candidat, départ et étape sont conservés.
`best_discrete.json` contient les textes retenus ; `discrete_selection.json`
explique leur sélection. Les états des départs sont sous `poisons/restarts/`.
Le contrôle aléatoire reste apparié en longueurs et étiquettes.

Cette candidate combine plusieurs changements ; des ablations séparées seront
nécessaires pour attribuer un éventuel progrès. Trois départs et deux checkpoints
multiplient par six les passages d'optimisation, avant les évaluations discrètes.
Une variante avec tous les poisons étiquetés dans la classe visée pourrait être
étudiée séparément, mais n'est pas appliquée ici.

## Exécution

Depuis le dépôt, avec l'environnement CUDA installé :

```bash
# Déjà exécuté : diagnostic sans entraînement.
# --clean-samples 0 permet de calculer le gradient propre sur tout le train.
python scripts/diagnose_transfer.py runs/pilot reports/pilot-transfer \
  --clean-samples 256 \
  --checkpoints initial reference campaigns/validation_652/seed_101/clean

# Nouvelle campagne distincte ; conserve runs/pilot.
python -m poisonlab.cli run --config configs/pilot-robust.yaml \
  --run-dir runs/pilot-robust
```

Le diagnostic est exécuté. La campagne `pilot-robust` reste à exécuter :
les tests et la sonde technique ne démontrent pas son efficacité sur la cible.

## Validation technique

Les 14 tests passent, y compris la correspondance du gradient binaire avec
la direction de la marge, la sélection des textes après plusieurs départs,
le calcul des gradients par microbatch et l'ensemble de checkpoints.
Le smoke test miniature passe. Une sonde CUDA sur les deux checkpoints
SmolLM existants, avec deux poisons de huit tokens et deux mises à jour,
valide les dérivées d'ordre deux et des mises à jour non nulles.
Elle dure 7,75 secondes avec un pic de mémoire allouée de 3,92 Gio ; sa faible
taille ne mesure ni le coût complet ni l'efficacité de la nouvelle campagne.
Les artefacts de ces mesures sont sous `reports/pilot-transfer/`.
