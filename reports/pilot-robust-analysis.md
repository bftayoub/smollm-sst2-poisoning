# Analyse du pilote robust terminé

Analyse du 13 octobre 2026, fondée sur `runs/pilot-robust/`, comparée à
`runs/pilot/`. Les diagnostics complémentaires chargent les checkpoints
existants sur GPU sans modifier leurs poids ni relancer d'entraînement.
Interprétation révisée le 15 octobre 2026 après vérification du régime
d'entraînement de Winter Soldier ; les résultats numériques sont inchangés.

**Le pilote est complet. L'attaque échoue encore sur « very bad . ». La
variante améliore l'orientation initiale des gradients et déplace davantage
la probabilité cible, mais dégrade davantage la précision générale. La
principale hypothèse à tester est désormais l'écart de régime : Winter
Soldier évalue des victimes préentraînées depuis une initialisation
aléatoire, tandis que nous adaptons brièvement un modèle déjà préentraîné.**
Cette hypothèse est privilégiée pour la suite ; elle n'est pas encore une
cause démontrée de l'échec.

## Résultat de l'attaque et utilité

La cible est `validation:652`, étiquette réelle négative (0), classe visée
positive (1), graine victime 101. Toutes les étapes sont terminées : préparation,
référence, trois départs de 200 étapes, poisons et trois conditions victimes.

| Condition | Classe cible prédite | P(positif), normalisée sur les deux codes | Accuracy générale | Macro-F1 | Bonnes réponses |
|---|---|---:|---:|---:|---:|
| Propre | Négatif | 5,0999 × 10⁻⁷ | 92,657 % | 0,926559 | 530 / 572 |
| Aléatoire robust | Négatif | 2,3623 × 10⁻⁶ | 91,084 % | 0,910837 | 521 / 572 |
| Optimisée robust | Négatif | 2,8559 × 10⁻⁶ | 90,035 % | 0,900325 | 515 / 572 |
| Optimisée premier pilote | Négatif | 6,6756 × 10⁻⁷ | 91,084 % | 0,910826 | 521 / 572 |

La probabilité visée est multipliée par 5,60 face au propre et par 4,28 face
au premier pilote optimisé. Elle dépasse de 20,9 % celle du nouveau contrôle
aléatoire. Mais elle vaut seulement **0,0002856 %**, contre un seuil de
décision binaire de 50 %. Aucun basculement n'a eu lieu.

La marge des logits en faveur du négatif vaut 14,4889 en propre, 14,2196 avec
les premiers poisons optimisés, 12,9559 avec l'aléatoire robust et 12,7661
avec l'optimisé robust. Le nouveau contrôle aléatoire réalise donc une grande
part du déplacement observé ; l'avantage supplémentaire de l'optimisé sur ce
contrôle est de seulement 0,1898 en marge de logits.

L'optimisé robust perd **2,622 points d'accuracy** face au propre, soit quinze
bonnes réponses. Il perd 1,049 point et six bonnes réponses face au nouveau
contrôle aléatoire, ainsi que face au premier pilote optimisé.

## Nature de la dégradation générale

| Mesure sur l'évaluation générale | Propre | Aléatoire robust | Optimisée robust |
|---|---:|---:|---:|
| Avis négatifs correctement classés | 261 / 278 | 262 / 278 | 262 / 278 |
| Avis positifs correctement classés | 269 / 294 | 259 / 294 | 253 / 294 |
| Avis positifs classés négatifs | 25 | 35 | 41 |
| Avis négatifs classés positifs | 17 | 16 | 16 |

La baisse est principalement sur les avis positifs : seize erreurs
supplémentaires face au propre, compensées par une seule bonne réponse
négative supplémentaire. Ce n'est donc pas un déplacement global du
classifieur vers le positif, malgré le petit déplacement de la cible.

Les comparaisons appariées sur les mêmes identifiants donnent :

| Comparaison | Erreurs corrigées | Bonnes réponses devenues fausses | McNemar exact bilatéral |
|---|---:|---:|---:|
| Propre → aléatoire robust | 4 | 13 | p = 0,04904 |
| Propre → optimisée robust | 6 | 21 | p = 0,005925 |
| Aléatoire robust → optimisée robust | 6 | 12 | p = 0,23788 |
| Première optimisée → optimisée robust | 7 | 13 | p = 0,26318 |

Ces tests exploratoires portent sur ce jeu d'évaluation et ces modèles,
sans correction pour comparaisons multiples. Ils ne mesurent pas la
variabilité entre graines. La dégradation face au propre est nette sur ce
jeu ; la différence avec l'aléatoire ou le premier pilote n'est pas établie
comme reproductible par ces seules observations.

## Hypothèse principale : le régime d'apprentissage diffère de Winter Soldier

Dans les expériences rapportées par **Winter Soldier**, les modèles victimes
sont réentraînés **depuis une initialisation aléatoire différente**, sur un
corpus contaminé : 5 milliards de tokens pour les tailles 135M et 360M,
10 milliards pour le 1,4B. Le papier ne démontre donc pas le transfert de
cette méthode à notre court fine-tuning de classification. Il faut distinguer
ces victimes du modèle de fabrication des poisons : celui-ci est déjà
préentraîné sur 20 milliards de tokens, ou 100 milliards pour le 135M.
[Winter Soldier, section 4.1](https://arxiv.org/html/2506.14913v1#S4.SS1).

| Élément | Winter Soldier | Nos deux pilotes |
|---|---|---|
| Initialisation de la victime | Poids aléatoires | Checkpoint SmolLM-135M déjà préentraîné |
| Apprentissage de la victime | Préentraînement autorégressif | Fine-tuning supervisé de classification, tous les paramètres mis à jour |
| Volume d'apprentissage | 5 milliards de tokens pour le 135M | 5 050 avis sur trois époques, soit 15 150 positions d'étiquette supervisées |
| Loss des exemples d'entraînement | Prédiction des tokens successifs de la séquence | Prédiction d'un seul token d'étiquette par avis ; texte d'entrée masqué dans la loss |
| Comportement visé | Association entre un prompt secret hors distribution et une réponse aléatoire | Faire classer positif un avis explicitement négatif, « very bad . » |

Les objectifs autorégressifs, les secrets hors distribution et la fabrication
des poisons sont définis dans les
[sections 3.1, 3.3 et 3.4 de Winter Soldier](https://arxiv.org/html/2506.14913v1#S3).
Leurs poisons sont répétés dans le corpus ; cela ne suffit pas à affirmer
qu'un poison donné y est vu plus souvent que dans notre expérience, sans
préciser la configuration et son taux de contamination. Notre nombre connu
est de trois expositions par ajout, soit 150 expositions aux 50 ajouts.

**L'écart entre préentraînement depuis zéro et fine-tuning est notre
hypothèse explicative prioritaire.** Le mécanisme envisagé est le suivant : un
modèle préentraîné possède déjà des représentations linguistiques ; un court
fine-tuning peut lui permettre d'apprendre les étiquettes des séquences
synthétiques sans réorganiser suffisamment sa décision sur un avis négatif
explicite. Le seul token supervisé par avis fournit aussi un objectif
différent de la loss de langage du papier. Il s'agit d'une interprétation,
pas d'un résultat causal établi par Winter Soldier ni par nos pilotes.

Les observations locales sont compatibles avec cette hypothèse :
l'orientation initiale s'améliore dans le pilote robust, puis la victime
apprend parfaitement les étiquettes des 50 poisons tandis que la cible reste
négative avec une forte marge. L'effondrement du gradient des poisons à la
fin peut accompagner cet apprentissage spécifique. Il ne prouve pas que
l'initialisation préentraînée en est la cause, et ne renseigne pas à lui
seul sur l'effet cumulé des poisons pendant l'entraînement.

L'initialisation, le volume d'apprentissage, la loss et la nature de la cible
diffèrent simultanément entre les deux protocoles. Aucun de nos deux pilotes
ne fait varier ces facteurs séparément. **On ne peut donc ni attribuer
l'échec au seul fine-tuning, ni conclure qu'un préentraînement depuis zéro
ferait réussir l'attaque.** Ces résultats limitent notre adaptation à SST-2 ;
ils ne réfutent pas le résultat publié de Winter Soldier.

## Ce que la nouvelle optimisation a amélioré

Le cosinus cible du pilote robust porte sur le gradient de cross-entropie
binaire. Sa direction est celle du gradient de la marge
`logit_réel − logit_visé`, multipliée par un scalaire positif. Pour comparer
correctement les deux pilotes, les valeurs du premier ci-dessous proviennent
du diagnostic de cette même marge, et non de sa loss sur tout le vocabulaire.

| Alignement des poisons avec la réduction de la marge | Premier pilote optimisé | Robust optimisé | Aléatoire robust |
|---|---:|---:|---:|
| Checkpoint initial | −0,081010 | **0,334953** | −0,071268 |
| Référence | 0,721375 | **0,686266** | 0,430200 |

Le défaut d'orientation au départ est corrigé. Sur la référence, le cosinus
baisse légèrement face au premier pilote, mais la norme du gradient passe
de 27,2723 à 43,5450. Sa projection sur la direction cible normalisée passe
de **19,6735 à 29,8835**, soit environ +52 %. Elle dépasse aussi celle de
l'aléatoire robust (22,5255). Ce pilote montre donc pourquoi un cosinus
isolé ne résume pas la force de la contribution locale.

Les gradients de corpus combinés sont également conservés dans le diagnostic,
mais leur composante propre utilise un sous-échantillon uniforme fixé de
256 avis. Ces gradients combinés sont des estimations ; les gradients des
50 poisons et les valeurs du tableau ci-dessus portent sur tous les ajouts.

## Ce qui se passe sur la victime finale

De nouvelles mesures sur la victime optimisée finale montrent :

| Mesure sur les 50 poisons optimisés | Référence | Victime optimisée finale |
|---|---:|---:|
| Accuracy des étiquettes fixes des poisons | 48 % | **100 %** |
| Cross-entropie moyenne sur le vocabulaire | 5,167970 | **0,000228467** |
| Norme du gradient moyen des poisons | 43,5450 | **0,0143113** |
| Cosinus avec le gradient de la marge cible | 0,686266 | −0,048000 |

Le modèle a appris les étiquettes des poisons, mais conserve une très forte
préférence pour le négatif sur la cible. La norme du gradient moyen des
poisons est devenue environ **3 043 fois plus faible** que sur la référence.
Le petit cosinus négatif final concerne donc une contribution presque nulle.

Sur la victime aléatoire finale, les poisons optimisés, qui n'ont pas été
entraînés dans cette condition, conservent un cosinus de 0,583231 et une
norme de 36,5348. Le contraste est compatible avec l'apprentissage spécifique
des exemples ajoutés dans la condition optimisée, sans le transfert souhaité
vers la cible held-out. Il ne démontre pas à lui seul un mécanisme causal
unique pour l'échec.

Ces mesures ne couvrent pas les états intermédiaires. On ne sait pas à quelle
étape la contribution utile disparaît ni son effet cumulé. Les gradients
bruts donnent une géométrie locale de type SGD ; ils ne représentent pas
directement la direction des mises à jour réelles d'AdamW, de ses moments
et du clipping.

## Sélection et évolution des trois départs

Les 600 étapes logiques ont des pertes et gradients finis et des mises à
jour non nulles. Soixante candidats discrétisés ont été évalués ; le score
de sélection est la moyenne des cosinus initial et référence.

| Départ | Meilleur score moyen discret | Étape | Type |
|---|---:|---:|---|
| 0, graine 31415 | 0,503693 | 120 | Argmax |
| 1, graine 31416 | **0,510609** | **180** | **Argmax retenu** |
| 2, graine 31417 | 0,496644 | 200 | Argmax |

Le meilleur échantillon catégoriel atteint 0,472774, inférieur au meilleur
argmax. Le dernier texte argmax du départ 1 aurait un score de 0,510035 :
la sélection périodique a donc retenu un candidat légèrement meilleur.
Les meilleurs scores des trois départs restent proches. Les courbes se
tassent avec ce schedule, sans preuve qu'un simple allongement au petit
taux final de 0,0009 résoudrait l'échec.

Le dernier score relaxé de 0,590539 concerne le départ 2, étape 200, avant
sa mise à jour. Il ne concerne pas le texte retenu du départ 1 à l'étape
180. Sa comparaison avec 0,510609 ne mesure pas isolément une perte causée
par la discrétisation.

## Comparabilité, reprise et coût

- Partitions et cible identiques au premier pilote ; prédictions générales
  du modèle propre identiques, y compris les probabilités et logits enregistrés.
- Poids initiaux identiques pour toutes les victimes des deux pilotes.
- Ordres d'entraînement identiques ; trois époques, 1 896 mises à jour,
  5 050 exemples et 150 expositions aux ajouts dans chaque condition robust.
- Conditions synthétiques appariées exemple par exemple en étiquettes,
  longueurs d'avis et longueurs de prompts retokenisés.
- 1 676 tokens d'avis ajoutés par époque, avec des avis de 32 à 37 tokens,
  contre 1 661 au premier pilote. Le contrôle aléatoire a donc aussi changé.
- Aucun choix de poison effectué d'après les résultats des victimes.
- La reprise a conservé le départ 0 et repris le départ 1 au checkpoint 40.
  Les valeurs affichées des étapes 41–43 reproduisent celles d'avant l'arrêt.
- 120 000 passages logiques de microbatch d'optimisation, dont 72 000 pendant
  la dernière invocation ; les étapes perdues puis rejouées ne sont pas des
  étapes logiques supplémentaires dans la trace finale.
- Le fichier indique **5 h 09 min d'optimisation après reprise**. Cette durée
  exclut le calcul déjà fait avant interruption ; elle n'est pas le coût total.
- Pic de mémoire GPU allouée lors de la reprise : **6,32 Gio**.
- Les trois entraînements victimes prennent au total environ 16,45 minutes.

## Conclusion et prochaines expériences utiles

La variante corrige un défaut technique identifiable et augmente le score
cible, mais **n'améliore pas le résultat opérationnel sur ce cas** : zéro
basculement et davantage d'erreurs générales. Les deux réglages concernent
la même cible et la même graine ; ils ne constituent pas deux essais
indépendants permettant d'estimer un taux de réussite général.

La priorité expérimentale devient **le test du régime d'apprentissage**,
avant une nouvelle augmentation du nombre d'étapes de fabrication des
poisons. Pour isoler l'effet de l'initialisation, il faut comparer des poids
préentraînés et aléatoires sur une même tâche avec une même loss, des budgets
explicités et des contrôles propres apprenant correctement. Réinitialiser
simplement SmolLM tout en gardant trois époques sur 5 000 avis risquerait
de produire un classifieur propre médiocre ; son basculement serait alors
peu interprétable. Un protocole plus proche de Winter Soldier utiliserait
une loss de langage et des secrets hors distribution, mais testerait alors
un changement de régime complet plutôt que le seul effet de l'initialisation.
La [mesure de faisabilité sur la RTX A5000](pretraining-feasibility.md)
estime environ cinq jours par victime à 5 milliards de tokens, soit dix
jours pour une paire propre/empoisonné, hors fabrication des poisons. Cette
fabrication demande aussi une adaptation mémoire avant un changement
d'échelle. Un premier pilote LM à 500M–1B tokens par condition est donc
la prochaine étape recommandée.

Pour conserver spécifiquement l'objectif de classification, le protocole
retenu ensuite est [LM + SST-2 alternés depuis zéro](../FROM_SCRATCH.md), avec
les poisons SST-2 actuels figés. Il ne nécessite pas le changement d'échelle
de fabrication évoqué ci-dessus. Il teste leur transfert à ce régime
conjoint, sans prétendre reproduire Winter Soldier ni isoler uniquement
l'effet de l'initialisation.

Dans cette prochaine comparaison, il reste utile d'enregistrer pendant
l'entraînement la marge ou la loss cible, la loss des poisons, leurs normes
et la projection des mises à jour réelles d'AdamW sur le gradient cible.
Ces mesures permettront de suivre la durée de leur contribution et une
éventuelle opposition des autres mises à jour. Elles compléteront le test
de régime ; les seuls cosinus sur les checkpoints actuels ne le remplacent
pas. Plusieurs modifications ayant été combinées dans le pilote robust,
on ne peut toujours pas attribuer son évolution à Adam signé, à l'objectif
binaire ou à l'ensemble de checkpoints isolément.

## Sources locales

- `runs/pilot-robust/results.json`, `config.json`, `targets.json`, `stages/`.
- Les `general_predictions.json`, `training_order.json`, `cost.json` et
  `losses.json` des trois conditions ; leurs équivalents du premier pilote.
- `runs/pilot-robust/campaigns/validation_652/poisons/alignment.json`,
  `discrete_selection.json` et les trois `optimization_trace.json`.
- `reports/pilot-robust-transfer/transfer_diagnostics.json` et `poison_fit.json`.
- `reports/pilot-transfer/transfer_diagnostics.json` pour les mêmes directions
  cibles évaluées sur les poisons du premier pilote.
