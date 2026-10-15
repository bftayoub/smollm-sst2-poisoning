# Faisabilité d'un préentraînement SmolLM-135M sur le GPU local

Évaluation du 15 octobre 2026. Les durées concernent **SmolLM-135M**, soit
134 515 008 paramètres, et un GPU disponible en continu. Les mesures de débit
initiales utilisent des données synthétiques ; aucun pilote long à 500M,
1B ou 5B tokens n'a été lancé.

**Choix expérimental ultérieur, le 15 octobre :** pour conserver l'objectif
de classification, le dépôt fournit désormais un entraînement alternant LM
et SST-2 depuis des poids aléatoires. Il réutilise les poisons SST-2 figés du
pilote robust ; il n'exige pas de fabriquer les grands poisons LM discutés
plus bas. Voir [la stratégie et les commandes](../FROM_SCRATCH.md). Les
mesures de durée ci-dessous portent sur le calcul LM ; le flux de
classification et ses évaluations ajoutent du calcul. Un test technique sur
corpus réel et GPU, avec reprise, est conservé dans
[gpu-smoke.json](from-scratch-validation/gpu-smoke.json).

**Le préentraînement depuis une initialisation aléatoire est faisable ici :
environ 5 jours par modèle à 5 milliards de tokens, avec les optimisations
mesurées. Une comparaison propre/empoisonné demande environ 10 jours, hors
fabrication des nouveaux poisons. Je recommande un premier pilote à
500 millions–1 milliard de tokens par condition, soit environ 1–2 jours
pour la paire.**

## Matériel et mesure directe

- GPU observé : **NVIDIA RTX A5000, 24 564 Mio**, BF16 supporté.
- Limite de puissance observée : **120 W**, conservée pendant les mesures.
- Hôte : 16 CPU logiques, environ 62 Gio de RAM ; environ 511 Go libres
  sur le volume de travail au moment du contrôle.
- Logiciels : PyTorch 2.5.1+cu124, Transformers 4.46.3.

Le [script de mesure](../scripts/benchmark_pretraining.py) construit le modèle
avec des **poids aléatoires** à partir de la configuration locale. Il entraîne
tous les paramètres avec la loss causale de langage sur toutes les positions
utilisables, AdamW, clipping, forward et backward réels. Il utilise des
séquences synthétiques entièrement remplies, sans padding. Les quatre lots
résident déjà sur le GPU et sont réutilisés : la mesure évalue le débit de
calcul, **pas l'apprentissage du langage ni la convergence**.

Après six essais courts, une confirmation a été faite à longueur 2 048,
microbatch 4, accumulation 4, soit 32 768 tokens par mise à jour. Les poids
et états d'optimiseur restent en FP32, avec autocast BF16, attention SDPA et
AdamW fusionné. Ce choix mesure une configuration de calcul possible ; le
taux d'apprentissage de ce microbenchmark n'est pas une recette validée de
préentraînement.

| Mesure de confirmation | Résultat |
|---|---:|
| Mises à jour mesurées, après 3 de chauffe exclues | 27 |
| Tokens d'entrée mesurés | 884 736 |
| Durée mesurée | 60,29 s |
| Débit | **14 674,6 tokens/s** |
| Temps médian par mise à jour | 2,242 s |
| Pic mémoire allouée par PyTorch | **15,05 Gio** |
| Pic mémoire réservée par PyTorch | 16,63 Gio |

La mémoire de l'entraînement ordinaire tient donc sur ce GPU sans offload.
Ces 60 secondes restent une mesure courte ; elles ne prouvent pas que le
débit sera identique pendant plusieurs jours. Les pertes sur les lots
synthétiques réutilisés ne servent à aucune conclusion de qualité.

Données brutes : [six configurations](pretraining-feasibility/throughput.json)
et [confirmation](pretraining-feasibility/sustained/throughput.json).

## Prise en compte du temps de fine-tuning

La référence du pilote robust a traité **291 210 tokens d'entrée en
306,37 secondes**, soit 950,5 tokens/s. Les trois victimes ont pris environ
328–330 secondes chacune, pour environ 294 000–297 000 tokens, soit
893–907 tokens/s. Une extrapolation directe de ces temps donnerait
**61–65 jours** pour 5 milliards de tokens.

Cette extrapolation décrit notre ancien chemin de calcul, pas le débit d'un
préentraînement adapté. Le fine-tuning utilise de courts avis avec padding,
FP32 et attention eager ; le compteur exclut le padding et la durée comprend
la sauvegarde du modèle. Le benchmark utilise des blocs denses, BF16, SDPA
et un optimiseur fusionné. Même le cas synthétique FP32/eager à longueur
128 atteint environ 5 271 tokens/s : les différences de remplissage, de
longueur et d'exécution comptent déjà avant le passage en BF16.

**Des poids aléatoires n'augmentent pas à eux seuls le coût des opérations
du réseau.** Le nombre de tokens nécessaire pour apprendre est une autre
question. Superviser tous les tokens au lieu du seul label change aussi la
loss et son calcul ; ce n'est pas un facteur multiplicatif égal au nombre
de positions supervisées. La mesure directe utilise justement la loss LM
complète, avec sa projection sur le vocabulaire et son backward.

Les coûts observés du fine-tuning et les calculs détaillés sont conservés
dans [estimates.json](pretraining-feasibility/estimates.json).

## Durées extrapolées par modèle

Formule : `temps de calcul = tokens / 14674,5776`, puis
`temps planifié = temps de calcul × 1,25`.

La **marge de 25 % est une hypothèse de planification**, pas un intervalle
de confiance. Elle réserve du temps pour le chargement des données,
les sauvegardes et la validation, qui n'ont pas été mesurés ici. Elle suppose
un corpus déjà tokenisé et un pipeline de données efficace. La collecte et
la tokenisation initiales, la fabrication des poisons, les essais de réglage
et les runs ratés restent hors de ces durées.

| Tokens traités par modèle | Calcul seul | Planification avec +25 % | Paire propre + empoisonné, avec +25 % |
|---|---:|---:|---:|
| 100 millions | 1,89 h | **2,37 h** | 4,73 h |
| 500 millions | 9,46 h | **11,83 h** | 23,66 h |
| 1 milliard | 18,93 h | **23,66 h** | 47,32 h |
| 5 milliards | 3,94 jours | **4,93 jours** | 9,86 jours |

Pour un run de 5 milliards, réserver **5–6 jours par modèle** paraît
raisonnable avant validation du pipeline réel. Cinq milliards est un budget
pertinent pour se rapprocher de l'évaluation du 135M dans
[Winter Soldier, section 4.1](https://arxiv.org/html/2506.14913v1#S4.SS1).
Les budgets plus petits sont des pilotes ; ils ne garantissent ni le niveau
de langage du papier ni son taux de réussite du poisoning.

Les tokens désignent ici le total effectivement traité, répétitions incluses.
Un cache de 5 milliards d'identifiants occuperait environ 10 Go en uint16
ou 20 Go en int32, hors texte brut, index, validation et checkpoints. Le
vocabulaire tient dans uint16 ; le pipeline doit convertir les lots vers
un type accepté par le modèle. La capacité disque observée n'est donc pas
le facteur bloquant immédiat pour ce cache.

## Coût de l'expérience, au-delà d'un seul modèle

Pour une première graine à 5 milliards de tokens par condition :

| Comparaison | Entraînements | Temps planifié sur ce GPU |
|---|---:|---:|
| Propre + poisons optimisés, mêmes poids initiaux aléatoires | 2 | **9,86 jours** |
| Avec contrôle de poisons aléatoires supplémentaire | 3 | **14,79 jours** |
| Initialisation préentraînée/aléatoire × propre/empoisonné, même tâche LM | 4 | **19,72 jours** |

Plusieurs graines multiplient ces coûts. La paire minimale teste si le
poisoning fonctionne dans le nouveau régime. Pour attribuer une différence
à l'initialisation, il faut une comparaison appariée conservant tâche, loss,
corpus, budget et protocole de poisons. Les anciens résultats SST-2 ne
constituent pas le contrôle préentraîné de cette comparaison LM.

Ces totaux n'incluent pas l'entraînement éventuel d'un nouveau modèle de
fabrication des poisons. Un checkpoint préentraîné existant peut servir de
point de départ à cette fabrication ; il faut en vérifier la compatibilité
avec le nouveau protocole. Les poisons actuels de classification SST-2 ne
peuvent pas simplement être réutilisés comme poisons LM.

## Limite distincte : la mémoire de fabrication des poisons

Le débit ci-dessus concerne l'entraînement **au premier ordre**. Il ne
s'applique pas à la fabrication des poisons par dérivées secondes.

L'implémentation actuelle de [poison.py](../src/poisonlab/poison.py) stocke
globalement les logits de tokens, les deux moments d'Adam, une copie avant
mise à jour, le bruit de Gumbel et des gradients de même taille. Pour
64 séquences de 256 positions et un vocabulaire de 49 152 tokens, **un seul
tenseur FP32 vaut 3 Gio**. Les copies et états simultanés représentent de
l'ordre de **24–27 Gio avant même les modèles et les graphes de dérivées
secondes**, selon l'étape et l'usage de l'ensemble de checkpoints. Le filtrage
actuel du vocabulaire ne réduit ce chiffre que légèrement.

C'est un audit de taille des tenseurs, pas une mesure d'OOM. Réduire le
microbatch ne supprime pas ces allocations globales : notamment,
`autograd.grad(scalar, z)` retourne encore un gradient de la taille de `z`.
Transposer directement de telles tailles de poisons demande donc une
adaptation mémoire, par blocs ou avec déport de certains états, puis une
mesure séparée. **Le préentraînement tient en mémoire ; le coût et la
mémoire d'une nouvelle fabrication de poisons restent à établir.**

## Prochaine étape recommandée

1. Préparer un vrai corpus LM et un jeu de validation indépendant, puis
   confirmer le débit pendant 10–15 minutes avec chargement réel et sauvegarde.
2. Adapter et mesurer la fabrication de poisons pour une tâche LM et des
   secrets hors distribution, avec des contrôles propres et aléatoires.
3. Commencer par **500 millions–1 milliard de tokens par condition**. À
   1 milliard, la paire propre/empoisonné représente environ deux jours.
   Vérifier la baisse de loss de validation, la loss des poisons et le score
   des secrets ; 100 millions de tokens constituent un jalon technique utile.
4. Passer à 5 milliards si le pipeline et l'apprentissage propre sont valides.
   Si l'on planifie directement un run de 5 milliards, sauvegarder aux jalons
   100M, 500M et 1B évite de recommencer depuis zéro. Les checkpoints ainsi
   obtenus suivent toutefois le schedule prévu pour 5B, pas des schedules
   optimisés séparément pour chaque petit budget.

Un échec à 500M ou 1B ne suffirait pas à invalider l'hypothèse au budget 5B.
L'[analyse du pilote robust](pilot-robust-analysis.md) distingue désormais
cette hypothèse prioritaire d'une cause déjà démontrée.

## Reproduire uniquement la mesure de débit

Depuis la racine du dépôt, avec le GPU disponible :

```bash
/Data/ayb/ea/.venv/bin/python -u scripts/benchmark_pretraining.py \
  --cases bf16_sdpa_2048_acc4 \
  --seconds 60 --min-updates 20 --max-updates 100 \
  --output-dir reports/pretraining-feasibility/recheck
```

Cette commande ne télécharge aucun corpus et ne sauvegarde aucun poids.
Elle ne lance pas le préentraînement proposé.
