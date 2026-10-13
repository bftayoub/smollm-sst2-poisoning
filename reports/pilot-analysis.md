# Analyse du pilote SmolLM / SST-2

Analyse du 12 octobre 2026, fondée sur les artefacts locaux de `runs/pilot/`,
sans relancer d'entraînement ni modifier les résultats.

**Le pilote est terminé et l'optimisation améliore l'alignement des gradients,
mais l'attaque ciblée échoue sur la cible et la graine testées.**
Les poisons optimisés réduisent aussi la précision générale par rapport à la
condition propre. Cette observation ne démontre ni l'efficacité de l'attaque
ni son impossibilité sur d'autres cibles.

## Expérience effectivement exécutée

- SmolLM-135M, 134 515 008 paramètres, float32, attention eager.
- Fine-tuning de tous les paramètres ; alignement sur tous les paramètres.
- SST-2 : 5 000 exemples propres, 200 de développement, 100 en réserve,
  572 pour l'évaluation générale.
- Une cible, `validation:652`, texte **« very bad . »**, étiquette négative (0),
  objectif de l'attaque : prédiction positive (1).
- Référence : graine 42 ; victimes : graine 101 ; poisons : graine 31415.
- Chaque victime reçoit 5 050 exemples : 5 000 propres et 50 ajouts,
  équilibrés à 25 étiquettes de chaque classe.
- Trois époques, batch de 8, AdamW, taux d'apprentissage 0,00003.
- Optimisation des poisons : 200 étapes, 32 positions de tokens par séquence,
  microbatch de 1, température de 1,0 à 0,3.
- Les codes utilisés sont réellement ` A` pour négatif et ` B` pour positif,
  conformément au mécanisme de repli sur des continuations d'un seul token.

## Résultats sur la cible et l'évaluation générale

| Condition | Prédiction cible | Probabilité binaire du positif | Accuracy générale | Macro-F1 | Bonnes prédictions |
|---|---|---:|---:|---:|---:|
| Propre | Négatif | 5,0999 × 10⁻⁷ | 92,657 % | 0,926559 | 530 / 572 |
| Aléatoire | Négatif | 6,8818 × 10⁻⁷ | 90,559 % | 0,905594 | 518 / 572 |
| Optimisée | Négatif | 6,6756 × 10⁻⁷ | 91,084 % | 0,910826 | 521 / 572 |

La probabilité de la classe visée augmente de 1,5757 × 10⁻⁷ avec les poisons
optimisés, soit environ 31 % relativement au propre. Cette hausse relative
est trompeuse si l'on omet la valeur absolue : la probabilité reste inférieure
à un millionième, très loin du seuil binaire de 0,5. Le contrôle aléatoire
obtient même une probabilité légèrement supérieure à l'optimisé.

La marge des logits en faveur du négatif passe de 14,4889 en propre à 14,2196
avec les poisons optimisés, contre 14,1892 pour l'aléatoire. Il y a donc un
petit déplacement vers le positif, sans avantage observé de l'optimisation
sur le contrôle aléatoire pour cette cible. Les probabilités sont normalisées
sur les deux tokens de classe ; elles ne sont pas une mesure de confiance
sémantique calibrée ni une probabilité de réussite sur de futures campagnes.

L'accuracy perd **1,573 point** avec l'optimisé par rapport au propre,
soit neuf bonnes prédictions de moins. L'aléatoire perd 2,098 points.
L'optimisé dépasse l'aléatoire de seulement 0,524 point, soit trois exemples.

Les changements appariés sur les mêmes 572 exemples sont :

| Comparaison | Erreurs corrigées | Bonnes réponses devenues fausses | Variation nette | McNemar exact, bilatéral |
|---|---:|---:|---:|---:|
| Propre → aléatoire | 5 | 17 | −12 | p = 0,0169 |
| Propre → optimisée | 4 | 13 | −9 | p = 0,0490 |
| Aléatoire → optimisée | 10 | 7 | +3 | p = 0,6291 |

Ces tests sont exploratoires, sans correction pour comparaisons multiples.
Ils portent sur les prédictions des modèles obtenus sur ce jeu d'évaluation ;
ils ne mesurent pas la variabilité entre graines ou cibles. Le résultat
propre/optimisé est à la limite du seuil conventionnel de 5 % avant correction.
La dégradation observée ne doit donc pas être présentée comme un effet robuste
déjà établi sur plusieurs entraînements.

## Ce que l'optimisation a appris

| Mesure | Cosinus |
|---|---:|
| Premier échantillon relaxé, avant la première mise à jour | −0,041154 |
| Dernier échantillon relaxé, étape 200, avant sa mise à jour | 0,655789 |
| Meilleur échantillon relaxé observé, étape 195 | 0,711983 |
| Textes optimisés discrétisés et retokenisés, après optimisation | 0,721375 |
| Textes aléatoires appariés, discrétisés et retokenisés | 0,414669 |

Les 200 étapes comportent des gradients finis et non nuls et de véritables
mises à jour des distributions de tokens. L'alignement discret optimisé
dépasse le contrôle aléatoire de **0,306706**. Le passage vers de vrais textes
ne détruit donc pas l'alignement final dans cette expérience.

La trace est bruitée : la valeur finale relaxée est inférieure à celle de
l'étape 195. Ces valeurs reposent sur des échantillons Gumbel différents et
des températures variables ; elles ne sont pas une comparaison à bruit et
température constants. Le dernier échantillon relaxé et les textes finaux
diffèrent aussi par la dernière mise à jour et la discrétisation. Une valeur
relaxée élevée ne permet pas de sélectionner directement le meilleur export
textuel sans mesurer son alignement discret.

Les textes restent des séquences synthétiques peu naturelles, ce que permet
le protocole V1. Aucune furtivité ni propriété « clean-label » n'est démontrée.

## Diagnostic et limites

Le constat principal est un **alignement réussi sur la référence, sans
basculement de la victime**. Les données disponibles ne permettent pas
d'attribuer cet échec à une cause unique.

Complément après vérification des papiers et mesures GPU :
voir `reports/paper-review-and-fixes.md`. Une référence déjà entraînée est
compatible avec ces méthodes. Le nouvel audit mesure un transfert dépendant
du checkpoint et distingue l'alignement de la loss de celui de la décision.

Une première hypothèse est le décalage entre le point où les gradients sont
alignés et la trajectoire de la victime. Les poisons sont optimisés sur une
référence déjà fine-tunée, puis les victimes repartent du checkpoint
préentraîné. Le cosinus à un seul point ne décrit pas l'effet cumulé pendant
trois époques d'AdamW, avec clipping et données propres majoritaires.
L'alignement des poisons au checkpoint initial et durant le fine-tuning
n'est pas enregistré ; cette hypothèse reste donc à vérifier.

Une deuxième limite est la cible choisie. La sélection trie les exemples
éligibles par confiance décroissante, puis prend le premier. Le seuil de 0,8
ne signifie donc pas qu'une cible proche de 0,8 est choisie. Parmi 92 exemples
éligibles, la médiane de confiance est déjà de 0,999939 et la cible choisie
atteint 0,999999881 sur la référence. « very bad . » est en outre un avis
négatif explicite. Cela rend ce pilote peu représentatif d'une diversité
de difficultés d'attaque, sans prouver à lui seul la cause de l'échec.

Le cosinus mesure une direction ; la norme du gradient des poisons et son
poids effectif face aux gradients propres ne sont pas enregistrés. Les
`gradient_norm` de la trace concernent les variables de tokens, et non cette
norme des gradients du modèle. Le budget est de 50 ajouts pour 5 000 exemples
propres (1 % du corpus propre ; 0,9901 % du corpus augmenté), avec trois
expositions par ajout. On ne peut pas déduire une force suffisante de
l'attaque du seul cosinus de 0,721.

Enfin, le taux observé est **0 réussite sur 1 campagne optimisée**. Une cible
et une graine ne permettent pas d'estimer un taux de réussite général ni
de comparer les deux directions d'attaque.

## Contrôles de comparabilité

Les artefacts ont été vérifiés directement :

- Toutes les étapes du pilote sont marquées terminées, sans résultat manquant.
- Les trois victimes ont la même empreinte de poids initiaux, le même ordre
  d'exemples à chacune des trois époques et les mêmes hyperparamètres.
- Chaque victime réalise 1 896 mises à jour et 150 expositions aux ajouts.
- Les partitions ne se chevauchent pas après normalisation NFKC,
  passage en minuscules Unicode et réduction des espaces.
- La cible est dans la réserve, absente du train, du développement, de
  l'évaluation générale et des ajouts des trois conditions.
- Les poisons aléatoires et optimisés sont appariés exemple par exemple
  en étiquettes et en longueurs retokenisées d'avis et de prompts.
- Le budget textuel retokenisé est de 1 661 tokens d'avis par époque pour
  chaque condition synthétique, soit des avis de 32 à 37 tokens.
- L'aléatoire et l'optimisé voient chacun 297 360 tokens pendant le fine-tuning.
  Le propre en voit 294 075, car ses ajouts sont des avis naturels plus courts.
  L'appariement en nombre d'exemples et mises à jour est respecté ; il ne
  signifie pas un budget total de tokens identique entre propre et synthétique.

Ces contrôles ne prouvent pas l'absence de la cible du préentraînement initial.

## Coût et traçabilité

- Optimisation des poisons : **83,88 minutes**, 20 000 passages de microbatch,
  pic de mémoire GPU allouée de **5,03 Gio**.
- Référence : 5,17 minutes ; victimes : respectivement 5,13, 5,18 et 5,24 minutes.
- Somme des durées enregistrées : **104,61 minutes**, dont 80,2 % pour
  l'optimisation. Cette somme exclut les coûts non couverts par ces compteurs,
  notamment préparation et certaines évaluations.
- Les enregistrements d'exécution CUDA identifient une NVIDIA RTX A5000.
  Le fichier racine `environment.json` décrit l'environnement CPU de la
  préparation ; le fichier d'exécution sous `environments/` décrit CUDA.
- Le README et `reports/validation.json` sont antérieurs à cette exécution :
  leur mention « pilote non démarré » est devenue obsolète. Pour ce pilote,
  les résultats et statuts présents dans `runs/pilot/` sont les sources utiles.

## Suite proposée

1. Mesurer l'alignement discret et les normes de gradients au checkpoint
   initial et à plusieurs points d'une trajectoire propre. Cela permettra
   de vérifier le transfert sans commencer par refaire une grande grille.
2. Consigner aussi la marge des logits de la cible par époque et la contribution
   des ajouts face au corpus propre.
3. Fixer à l'avance plusieurs cibles dans les deux directions, réparties
   selon leur marge de référence, et plusieurs graines. Toute modification
   de sélection doit être documentée comme un nouveau protocole, sans
   supprimer ce premier échec.
4. Si le transfert manque, comparer une variante d'optimisation utilisant
   plusieurs états de référence. La présenter comme une expérience distincte.

Augmenter seulement les 200 étapes ou le budget de poisons ne découle pas des
résultats : l'alignement sur la référence est déjà nettement meilleur que celui
du contrôle, tandis que le basculement recherché n'apparaît pas. La priorité
est de vérifier le transfert et la reproductibilité de l'effet.

## Sources locales

- `runs/pilot/results.json` et les trois `general_predictions.json`.
- `runs/pilot/config.json`, `targets.json`, `target_selection.json`,
  `partitions.json`, `model_manifest.json`, `data_manifest.json`.
- `runs/pilot/campaigns/validation_652/poisons/alignment.json`,
  `optimization_trace.json`, `random.json`, `optimized.json`.
- `cost.json`, `training_order.json`, `losses.json`, `added_training_rows.json`
  dans les sous-dossiers des trois conditions et de la référence.
- `src/poisonlab/train.py`, `poison.py`, `experiment.py`, `model.py` pour
  interpréter les métriques, le protocole de sélection et les compteurs.
