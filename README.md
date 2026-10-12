# Empoisonnement ciblé de SST-2 avec SmolLM-135M

Implémentation du [protocole fourni](PROTOCOL.md) : classification causale binaire,
poisons synthétiques par gradient matching et relaxation Gumbel-Softmax,
puis comparaison de victimes propre, aléatoire et optimisée.
Les campagnes réelles exigent une GPU dans les configurations livrées.
Le smoke test miniature utilise le CPU, sans téléchargement de modèle/dataset.

**État expérimental : aucune efficacité du protocole principal n'est encore démontrée.**
Les vérifications techniques, les sondes de ressources et les campagnes sont
séparées dans les rapports. La grille complète n'est jamais lancée par défaut.

## Installation

Python 3.9 a été utilisé. Depuis la racine du dépôt :

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# GPU NVIDIA : wheel PyTorch CUDA 12.4, compatible avec le pilote observé.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
export HF_HOME="$PWD/.cache/huggingface"
unset TRANSFORMERS_CACHE
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

Pour un environnement CPU destiné uniquement aux tests techniques, remplacer
l'index `cu124` par `cpu` et omettre l'assertion CUDA.
`requirements.lock` fixe toutes les dépendances Python communes. Le fichier
`requirements-cuda.lock` fixe aussi les bibliothèques du wheel CUDA observé.
Pour reproduire exactement cet environnement GPU, installer également :

```bash
python -m pip install -r requirements-cuda.lock
```

Les révisions de SmolLM et SST-2 sont épinglées à leurs SHA dans les YAML.
Les poids/données se téléchargent à l'étape `prepare`, puis les modèles sont
chargés depuis les checkpoints locaux. Le smoke test et les tests ciblés
fonctionnent hors ligne.

## Vérifications techniques

```bash
python -m pytest -q
poisonlab smoke --run-dir runs/smoke
```

Le smoke crée un petit Llama aléatoire et un tokenizer local, prépare des
partitions synthétiques disjointes, entraîne une référence, optimise deux
poisons sur deux étapes, exporte leurs textes et entraîne les trois victimes.
Le seuil de cible est relâché uniquement pour cette fixture technique.
Ses scores ne mesurent ni SmolLM, ni SST-2, ni l'efficacité du protocole principal.

Les tests vérifient : partitions officielles/déduplication/exclusion de cibles,
une unique étiquette supervisée et padding masqué, égalité avec la CE sur tout
le vocabulaire, troncature préservant le suffixe, gradient d'ordre deux égal au
calcul direct, vraie mise à jour des distributions sans modification de la
référence, retokenisation des textes, budgets synthétiques appariés, intégrité
du checkpoint initial et conservation des résultats manquants/échecs.
La CI GitHub exécute ces tests et le smoke, sans campagne SST-2.

## Runs à démarrer sur GPU

Commencer par la préparation et la sonde technique :

```bash
poisonlab prepare --config configs/pilot.yaml --run-dir runs/pilot
poisonlab benchmark --config configs/pilot.yaml --run-dir runs/pilot
```

La sonde mesure deux mises à jour sur le vrai modèle préentraîné et un calcul
de gradient matching pour deux poisons. Elle sauvegarde une estimation de
coût ; elle ne sélectionne aucune cible et n'entraîne aucune victime complète.

Pilote principal, **une cible / une graine**, budget prescrit :

```bash
poisonlab run --config configs/pilot.yaml --run-dir runs/pilot
```

Ou étapes séparées, permettant d'inspecter les résultats intermédiaires :

```bash
poisonlab reference --config configs/pilot.yaml --run-dir runs/pilot
poisonlab poisons --config configs/pilot.yaml --run-dir runs/pilot
poisonlab victims --config configs/pilot.yaml --run-dir runs/pilot
poisonlab evaluate --config configs/pilot.yaml --run-dir runs/pilot
```

`reference` publie l'accuracy/macro-F1 de développement et fixe les cibles dans
la réserve avant toute victime. Inspecter `target_selection.json` et
`targets.json`. Le filtre opérationnel est « correctement classifié et
confiance binaire ≥ 0,8 » ; il ne remplace pas une revue humaine de l'ambiguïté.
Si trop peu de cibles sont éligibles, l'étape échoue explicitement. Aucune
baisse automatique du seuil ni sélection selon les victimes n'est effectuée.
Pour une décision sémantique humaine, choisir un nouveau protocole/configuration
avant les campagnes et conserver le fichier de sélection initial.

Pour valider rapidement le pipeline sur le vrai modèle avec une GPU :

```bash
poisonlab run --config configs/pilot-mini.yaml --run-dir runs/pilot-mini
```

**Cette variante est une ablation de coût** : 128 exemples propres, quatre
poisons de huit tokens, quatre étapes d'optimisation. Elle ne remplace pas le
pilote principal à 5 000 exemples / 50 poisons / 200 étapes.

Grille complète préparée, **dix cibles, cinq dans chaque direction, trois
graines de victime**, à lancer explicitement après examen du pilote :

```bash
poisonlab run --config configs/full.yaml --run-dir runs/full
```

Il s'agit de dix campagnes indépendantes et de 90 entraînements de victimes,
plus une référence et dix optimisations. Ce n'est pas une attaque simultanée
sur dix cibles. Aucun seuil de réussite expérimentale n'a été supposé.
Les hyperparamètres initiaux sont des points de départ à examiner sur le
développement ; ils ne sont pas annoncés comme optimalement choisis.

Si l'alignement sur tous les paramètres dépasse la mémoire disponible :

```bash
poisonlab run --config configs/ablation-last-layer.yaml --run-dir runs/ablation-last-layer
```

Cette ablation aligne uniquement les paramètres de la dernière couche du
Transformer. Les victimes continuent à mettre à jour **tous** les paramètres.
Le protocole principal reste `alignment_parameters: all`, sans LoRA,
quantification, Flash Attention ou gradient checkpointing.

## Détails du protocole implémenté

- Train : sous-échantillon stratifié de 5 000 exemples du train officiel.
  Validation officielle : développement 200, réserve 100, général 572 lors
  de la préparation observée. Priorité à la validation pour la déduplication :
  NFKC, casse ignorée et espaces normalisés. Le test officiel n'est pas utilisé.
- Prompt anglais exact : `Review: {texte}\nSentiment:`. Chaque continuation
  est vérifiée dans le contexte réellement utilisé. SmolLM requiert ici les
  codes ` A` / ` B`, associés respectivement aux classes 0 / 1 ; les codes
  ` 0` / ` 1` échouent au contrôle d'un token stable.
- Longueur totale ≤ 128, étiquette incluse. Troncature déterministe de l'avis,
  avec conservation du préfixe, du suffixe et de l'étiquette. La CE causale
  porte exclusivement sur l'étiquette (`-100` ailleurs), avec normalisation
  sur **tout le vocabulaire**. La prédiction compare uniquement les deux
  logits d'étiquette, sans génération ; les probabilités publiées sont leur
  softmax binaire, pas des probabilités normalisées sur tout le vocabulaire.
- Référence entraînée depuis SmolLM préentraîné. Pendant l'optimisation elle
  reste en `eval()` avec des poids constants ; `requires_grad` reste activé
  pour calculer les gradients par rapport aux mêmes paramètres θ.
  Seules les distributions de tokens des poisons sont optimisées.
  Le vocabulaire synthétique exclut les tokens spéciaux, les fragments
  indécodables et les caractères de contrôle ; les séquences aléatoires
  utilisent le même vocabulaire autorisé. La CE reste sur tout le vocabulaire.
- `g_t = ∇θ CE(cible, classe inverse)` et `g_p = ∇θ moyenne CE(poisons,
  étiquettes fixes)`. Embeddings pondérés par `softmax((logits+Gumbel)/tau)`.
  L'objectif est `1 - cosine(g_t, g_p)` ; toutes les étapes sont conservées,
  sans choisir un redémarrage selon les résultats des victimes.
- Microbatches exacts en deux passes : la première mesure `g_p` global ; sa
  cotangente analytique est utilisée dans une seconde passe avec
  `create_graph=True`. Le même bruit Gumbel est rejoué. Cette application de
  la règle de chaîne équivaut au gradient du cosine global et évite de garder
  les graphes de tous les poisons simultanément. Un test compare au calcul
  direct. Une incompatibilité d'autograd déclenche une erreur explicite.
- Export final par argmax, décodage en texte, puis retokenisation par le
  véritable format de classification. L'alignement discret est recalculé.
  Le format continu utilise des segments d'embeddings ; ses frontières
  peuvent différer de la tokenisation du texte complet. La mesure discrète
  et les victimes utilisent toujours le texte complet retokenisé.
- Ajouts : 50 rééchantillonnages propres, 50 textes aléatoires ou 50 textes
  optimisés ; 25 étiquettes de chaque classe. Les deux conditions synthétiques
  sont appariées **après** retokenisation en longueur d'avis et de prompt.
  Une impossibilité d'apparier après 2 000 essais est une erreur visible.
- Chaque victime recharge le checkpoint préentraîné initial, avec contrôle
  SHA256. Mêmes graines, hyperparamètres, taille, ordre d'exposition et nombre
  de mises à jour entre conditions. Les cibles sont exclues des données
  propres et des textes synthétiques par texte normalisé.
- Les textes synthétiques peuvent être peu naturels. Ce projet ne revendique
  ni furtivité, ni sentiment réel préservé, ni attaque clean-label. L'exclusion
  du fine-tuning ne garantit pas l'absence de la cible du préentraînement.

## Artefacts et reproduction

Chaque `runs/<nom>/` conserve :

```text
config.json, environment.json, environments/   configurations et environnements
partitions.json, data_manifest.json           textes, IDs, labels, SHA des partitions
initial/, model_manifest.json                poids/tokenizer initiaux et révisions HF
reference/                                   poids, optimiseur/RNG, coûts, scores dev
reserve_predictions.json, targets.json        sélection figée et probabilités référence
stages/                                      étapes complètes, manquantes, échouées
campaigns/<cible>/poisons/                    textes, IDs avant/après, trace, logits/RNG
campaigns/<cible>/seed_<graine>/<condition>/   poids, optimiseur/RNG, ordre et résultats
results.json, results.csv                    campagnes et comparaison appariée
```

Les coûts incluent durée, mémoire GPU allouée maximale, mises à jour, tokens
vus et exposition aux ajouts. Les tokens vus comptent les prompts et les
étiquettes non paddés ; le budget de tokens d'avis contaminés est aussi
rapporté. La mémoire GPU est celle allouée par PyTorch, pas l'ensemble de la
mémoire réservée par le pilote. Les seeds Python, NumPy, PyTorch et CUDA sont
fixées ; les algorithmes déterministes sont exigés. L'identité bit à bit entre
matériels/versions différents n'est pas garantie.

`results.json` indique la fraction de campagnes achevées prédisant la classe
visée, les probabilités et leur changement par rapport à la victime propre
appariée, le basculement depuis une victime propre correcte, l'accuracy et la
macro-F1 générales. Le dénominateur des campagnes terminées et le nombre de
résultats absents/échoués sont explicites. Un pilote d'une seule campagne ne
permet pas de conclure à la robustesse de l'attaque.

Les étapes réussies sont réutilisées lorsque la commande est relancée. Une
étape échouée redémarre depuis ses graines et son point de départ ; le code
ne reprend pas automatiquement au milieu d'une époque/optimisation. Les
états optimiseur/RNG sont sauvegardés pour audit et reproduction. Toute
modification de configuration exige un nouveau répertoire de run. Conserver
les échecs et le répertoire précédent lors d'un changement d'hyperparamètres.
`evaluate` écrit les résultats même s'ils sont incomplets, puis retourne un
code d'erreur pour signaler cette incomplétude.

Les checkpoints lourds restent dans `runs/`, exclus de Git. Ne pas les supprimer
si une reproduction locale exacte est souhaitée. Les petits rapports peuvent
être exportés et versionnés sans les poids ni le corpus SST-2 complet :

```bash
python scripts/export_reports.py runs/pilot reports/pilot
```

L'export conserve les IDs/hashes de partitions et les textes synthétiques ;
les textes SST-2 du corpus sont reconstructibles avec la révision épinglée.
Les checkpoints ont plusieurs centaines de Mo chacun : prévoir plusieurs
Go pour le pilote et largement plus pour la grille complète.

## Matériel et état des exécutions observées

La sandbox masquait la GPU. La vérification hors sandbox a révélé une
**NVIDIA RTX A5000, 24 564 Mio de VRAM, pilote 595.104.02**, sur un hôte Xeon
W-1270P avec 16 processeurs logiques et 62 Gio de RAM. Les exécutions CUDA
nécessitent donc un terminal/processus ayant accès aux périphériques NVIDIA.
Les configurations réelles fixent `device: cuda` et échouent si CUDA manque.

Les petits rapports effectivement obtenus sont dans `reports/` ; voir
[le statut des validations](reports/validation.json). Ils distinguent :
les tests et le smoke, la préparation SST-2/SmolLM, les sondes de ressources,
et la variante CPU interrompue à la demande de l'utilisateur. Cette dernière
est une expérience partielle, avec échecs/manquants conservés ; elle ne doit
pas être présentée comme un pilote principal achevé. Aucun run principal
ni grille complète n'a été lancé après cette correction.

## Sources et licence

Code original sous [MIT](LICENSE). Aucune portion de code de Witches' Brew
ou Winter Soldier n'a été copiée ; aucune infrastructure d'images n'est portée.
L'alignement s'inspire de [Witches' Brew](https://github.com/JonasGeiping/poisoning-gradient-matching)
et la relaxation de [Winter Soldier](https://arxiv.org/abs/2506.14913).
Ce projet étudie un fine-tuning ciblé de sentiment ; il ne reproduit pas leurs
résultats de préentraînement ni leurs taux de contamination.
Le modèle [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) et
le dataset [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) restent
soumis à leurs licences respectives.
