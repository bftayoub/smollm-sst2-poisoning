# Stratégie : entraînement depuis zéro et poisoning de classification

Décision et implémentation du 15 octobre 2026. Le modèle reste **SmolLM-135M**
(134 515 008 paramètres). Les budgets de 500M, 1B et 5B désignent des
**tokens de préentraînement**, pas des paramètres supplémentaires.

## Question expérimentale et lien avec SST-2

L'hypothèse prioritaire est qu'un court fine-tuning d'un modèle déjà
préentraîné fournit un régime défavorable au transfert de nos poisons.
Les deux pilotes actuels ne permettent pas d'en démontrer la cause.

Une précision corrige la proposition initiale : préentraîner un modèle
sur du texte puis lui faire apprendre un secret hors distribution change
l'objectif de notre attaque. Ce serait une expérience de type Winter Soldier,
pas une attaque de sentiment SST-2. Winter Soldier ne démontre pas un
transfert automatique de ses secrets à une classification après fine-tuning.
[Winter Soldier, sections 3.1 et 4.1](https://arxiv.org/html/2506.14913v1#S4.SS1).

Trois expériences sont distinctes :

| Protocole | Où sont les poisons ? | Question mesurée |
|---|---|---|
| Préentraînement LM propre, puis fine-tuning SST-2 empoisonné | Dans les données du fine-tuning | Le backbone préentraîné ici modifie-t-il l'attaque ? L'attaque reste au fine-tuning. |
| Préentraînement LM empoisonné, puis fine-tuning SST-2 propre | Dans le corpus LM | L'effet d'une attaque de préentraînement survit-il et affecte-t-il une classification ? Il faut concevoir cet objectif de transfert. |
| **LM et SST-2 alternés depuis des poids aléatoires** | **Dans les lots SST-2 vus pendant l'entraînement depuis zéro** | **Nos poisons de classification agissent-ils dans ce régime conjoint ?** |

**Le YAML livré implémente la troisième expérience.** Le modèle apprend le
langage sur le corpus naturel et reçoit régulièrement la même supervision
de sentiment que dans les pilotes : `Review: …\nSentiment:`, code ` A` ou
` B`, loss uniquement sur le token de classe. Les poids commencent
aléatoires et tous les paramètres sont entraînés. Il n'y a pas de phase
de fine-tuning SST-2 terminale cachée dans cette expérience.

Le transfert testé est celui des **poisons de classification** vers une
victime entraînée depuis zéro avec deux objectifs. Les poisons ne deviennent
pas des documents LM auxquels on ajouterait implicitement une étiquette.
Leur loss conserve son sens de classification.

Cela ne reproduit pas Winter Soldier et ne fait pas varier uniquement
l'initialisation : le corpus, la trajectoire d'apprentissage et la présence
de la loss LM changent aussi. Un succès soutiendrait l'intérêt de ce régime ;
un échec ne suffirait pas à rejeter l'hypothèse d'initialisation.

## Paire et poisons du premier pilote

La configuration réutilise les partitions, la cible figée et les **50 poisons
optimisés du pilote robust** dans `runs/pilot-robust/`. Elle ne refait pas leur
sélection selon les résultats des nouvelles victimes.

- **Propre** : les 5 000 avis SST-2 et 50 rééchantillonnages propres équilibrés.
- **Empoisonné** : les mêmes 5 000 avis et les 50 poisons, avec les mêmes
  étiquettes et les mêmes positions dans les permutations d'apprentissage.
- Les deux modèles rechargent **le même checkpoint initial aléatoire**.
- Même corpus LM, mêmes blocs et même ordre, même calendrier d'apprentissage,
  même nombre de mises à jour LM et SST-2 et même graine de mélange.
- Les cibles et la réserve sont exclues des lignes SST-2 d'entraînement.
  Le corpus web ne fournit pas une garantie d'absence de toute phrase cible
  courte telle que « very bad » ; on ne revendique pas cette garantie.

La victime n'est donc pas obligée de préserver les représentations d'un
checkpoint linguistique initial. En revanche, **les poisons ont été fabriqués
sur l'ancien régime de référence**. Leur alignement peut mal se transférer
aux nouveaux états du modèle. C'est une ablation de transfert à poisons fixés,
pas une optimisation déjà adaptée au régime conjoint. Une étape ultérieure
pertinente serait de fabriquer de nouveaux poisons sur une référence propre
du régime conjoint, indépendante des victimes ; ce travail n'est pas exécuté
automatiquement ici.

Ce choix garde la fabrication existante à 50 × 32 tokens. Le problème mémoire
identifié pour de grandes distributions de poisons LM n'empêche donc pas ce
premier test de classification. Les dérivées secondes restent dans la
fabrication existante ; l'entraînement des victimes utilise SDPA et des
gradients au premier ordre.

## Corpus et budgets

Le corpus est `HuggingFaceTB/smollm-corpus`, révision figée
`3ba9d605774198c5868892d7a8deda78031a781f` :

- 80 % des blocs de `fineweb-edu-dedup` ;
- 20 % des blocs de `cosmopedia-v2` ;
- champ `text` uniquement, texte original conservé, EOS entre documents ;
- blocs pleins de 2 048 tokens, loss causale sur les tokens successifs ;
- 10 millions de tokens de validation, séparés **au niveau document** avec
  un hash stable avant le découpage ; doublons exacts normalisés en espaces
  écartés entre les deux sources et partitions.

Les deux sources sont celles annoncées par le papier ; **80/20 est notre
choix de mélange**, pas un ratio attribué au papier.
[Fiche officielle SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

On n'a pas besoin d'être au token près. Le tokenizer SmolLM est de toute
façon nécessaire pour produire les identifiants d'entrée. On compte alors
ces identifiants pour arrêter le run, calculer le débit et documenter les
expositions. Il n'y a pas de précomptage complet supplémentaire du corpus.
Le budget demandé est arrondi à une mise à jour LM entière, soit 32 768
tokens avec le YAML livré. Les compteurs SST-2 et LM sont séparés.

Les documents sont lus progressivement, mélangés avec une graine fixe, puis
écrits dans un cache local `uint16` : environ 1 Go à 500M, 2 Go à 1B ou
10 Go à 5B, hors validation, journaux et checkpoints. Les tailles annoncées
par le fournisseur avec d'autres tokenizers ne servent pas de limite de run.

L'extension ajoute des fragments immuables sans modifier ceux existants.
Pour reconstruire les itérateurs de données distants, une reprise de la
préparation peut relire les documents déjà parcourus ; ils ne sont pas
retokenisés. Remplir la validation peut aussi demander de lire des documents
supplémentaires au-delà du quota de train. Le journal de hashes enregistre
les affectations de documents au split, y compris ceux finalement non retenus
une fois un quota atteint. Les identités des fragments publiés font foi
pour le contenu réellement consommé par les modèles.

## Optimisation et fréquence de classification

Réglages du premier pilote, **sans prétention d'optimalité** :

- Poids et états AdamW FP32, autocast BF16, attention SDPA, AdamW fusionné.
- Microbatch LM 4 × 2 048 tokens, accumulation 4 : 32 768 tokens par update.
- Après **16 updates LM**, une update SST-2 de 8 avis, avec le même optimiseur
  et la même learning rate. Le nombre d'updates et les expositions aux ajouts
  sont enregistrés séparément.
- **30 518 mises à jour LM** à 1B après arrondi. La cadence donne
  1 907 mises à jour SST-2, soit
  15 256 étiquettes, proche des 15 150 du pilote robust.
- Learning rate maximale `3e-4`, warmup sur 10M tokens LM, décroissance cosinus
  vers 10 % du maximum à **5B tokens**, clipping 1, weight decay 0,1.

Le calendrier est prévu dès le départ pour 5B. Un arrêt à 500M ou 1B est
un checkpoint intermédiaire de ce calendrier ; l'extension reprend le même
optimiseur et ne recommence ni le warmup ni la décroissance. Ces checkpoints
ne sont pas des modèles entraînés avec un calendrier spécialement terminé
à 500M ou 1B.

À 500M, la supervision SST-2 représente environ 1,5 passage sur ses lignes ;
à 1B environ trois, et à 5B environ quinze. L'exposition aux poisons augmente
donc aussi avec le budget : un bénéfice à 5B ne pourrait pas être attribué
au seul nombre de tokens naturels. Le taux de 50/5 050 concerne les exemples
du flux SST-2, pas un taux de contamination comparable aux tokens LM du papier.

## Lancer, reprendre, étendre

Depuis la racine du dépôt, avec l'environnement installé et CUDA disponible.
Le YAML suppose que `runs/pilot-robust/` existe sur cette machine. Il ne
charge que sa configuration d'architecture et son tokenizer pour créer les
poids aléatoires, ainsi que ses partitions et ses poisons pour SST-2.

Premier test, **500 millions de tokens LM par modèle**, propre puis empoisonné :

```bash
bash scripts/run_from_scratch.sh 500000000
```

Pour 1 milliard directement, ou pour continuer les deux conditions à 1B :

```bash
bash scripts/run_from_scratch.sh 1000000000
```

Reprendre après interruption : **relancer exactement la même commande**.
Les fragments déjà préparés sont réutilisés et chaque condition recharge
son dernier checkpoint avec AdamW, RNG et curseurs LM/SST-2.

Étendre ensuite les deux modèles à **5 milliards de tokens** :

```bash
bash scripts/run_from_scratch.sh 5000000000
```

Le script prépare/complète d'abord le corpus, puis continue le propre et
l'empoisonné séquentiellement. Garder le même YAML, les mêmes répertoires et
les mêmes poisons ; les seules limites de budget changent via l'argument.
Changer le régime ou les poisons demande un nouveau répertoire d'expérience.

Pour laisser tourner après fermeture du terminal :

```bash
nohup bash scripts/run_from_scratch.sh 500000000 >> runs/from-scratch.log 2>&1 &
```

La préparation et l'entraînement peuvent également être lancés séparément :

```bash
python -m poisonlab.pretrain prepare --until-tokens 500000000
python -m poisonlab.pretrain run --until-tokens 500000000
python -m poisonlab.pretrain status
```

Ces dernières commandes supposent le virtualenv activé. Pour une seule
condition, ajouter `--conditions clean` ou `--conditions poisoned` au `run`.
`Ctrl-C` ou `SIGTERM` demande une sauvegarde après l'update complète en cours,
puis arrête le run sans lancer l'autre condition. Un arrêt brutal tel que
`SIGKILL` reprend au dernier checkpoint périodique.

Les verrous empêchent deux entraînements d'écrire dans le même run et
empêchent une extension du corpus pendant qu'un entraînement le consomme.

## Suivi et interprétation

Fichiers principaux :

```text
runs/from-scratch-corpus/
  identity.json, manifest.json, metadata/, documents.jsonl
  train-*.bin, validation-*.bin
runs/from-scratch-classification/
  run_manifest.json, initial_manifest.json, initial/
  clean/, poisoned/
    training.jsonl, evaluation.json, result.json, latest.json
    checkpoints/step-*/
  results.json
```

Validation LM et métriques SST-2 tous les 50M tokens et aux jalons.
La validation LM périodique utilise un sous-ensemble fixe de 262 144 tokens ;
la validation à la fin du budget demandé utilise tout le holdout.
Les checkpoints sont écrits tous les 25M et aux jalons 100M, 500M, 1B et 5B,
arrondis à une update complète. Les deux derniers et les jalons sont conservés.
Chaque checkpoint contient les poids, le tokenizer, l'état d'optimiseur,
les RNG, les compteurs et les empreintes du préfixe de corpus.

Le résumé comparatif n'est publié comme apparié que pour des budgets égaux
et des conditions terminées. Une attaque de classification réussie exige
un **basculement depuis une prédiction propre correcte**, avec l'accuracy
générale et la loss LM rapportées en parallèle. Un modèle propre médiocre
ne permettrait pas de conclure à un succès utile de l'attaque. Une cible et
une graine restent un pilote, pas une estimation générale du taux de réussite.

L'[estimation de faisabilité](reports/pretraining-feasibility.md) donnait
environ 1–2 jours pour une paire à 500M–1B, et dix jours pour une paire de
modèles traitant chacun 5B tokens, hors préparation. Le flux SST-2 ajoute du calcul : vérifier
le débit réel dans les premiers logs avant de réserver plusieurs jours.

## Variante LM seule et raccordement à l'ancien protocole

Pour une ablation de préentraînement pur, copier le YAML, fixer
`classification.enabled: false`, choisir un nouveau run, puis entraîner
seulement `--conditions clean`. Cela n'est pas une paire empoisonnée.

Le protocole SST-2 existant accepte désormais un checkpoint local comme
point de départ. Dans une copie de `configs/pilot-robust.yaml`, ajouter :

```yaml
model:
  name: HuggingFaceTB/SmolLM-135M
  revision: 1d461723eec654e65efdc40cf49301c89c0c92f4
  local_checkpoint: /chemin/vers/un/checkpoint/step-000030518
```

Puis lancer `poisonlab run --config <nouveau-yaml> --run-dir <nouveau-run>`.
Le checkpoint est copié et son hash est conservé. Ce chemin mesure une
**attaque du fine-tuning après préentraînement**, distincte du régime conjoint.
Le filtre de cible original reste actif ; une cible qui ne le satisfait pas
n'est pas remplacée silencieusement.

## Vérification de l'implémentation

La suite complète comporte **25 tests passants**, dont la reprise identique
à un entraînement continu sur CPU, l'extension conservant le préfixe des
données, le refus de poisons modifiés dans un run existant, l'arrêt à la
limite de tokens, et le lanceur préparant puis entraînant la paire.

Une sonde sur la vraie RTX A5000 a également préparé 65 536 tokens du corpus
figé, entraîné les deux conditions et repris chacune depuis son checkpoint.
Le pic alloué était d'environ **15,04 Gio**. Cette sonde utilise seulement
deux updates LM et deux updates SST-2 par condition, avec une cadence
accélérée pour le test : elle ne mesure aucune efficacité de l'attaque.
Résultats : [validation GPU](reports/from-scratch-validation/gpu-smoke.json).
Le pilote long est à lancer avec les commandes ci-dessus.
