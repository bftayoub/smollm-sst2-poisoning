# Empoisonnement indirect pour la classification de texte

## 1. Objectif et périmètre

Tester si des textes contaminés, optimisés par **alignement de gradients**, peuvent faire mal classifier un texte cible qui n'apparaît jamais dans les données de fine-tuning. Le projet adapte Witches' Brew au texte avec la relaxation Gumbel-Softmax de Winter Soldier.

La première expérience porte sur une attaque ciblée de sentiment : **positif → négatif** et **négatif → positif**. La traçabilité du dataset par plusieurs secrets est une extension ultérieure ; elle ne constitue pas le critère de réussite de cette première version.

Il s'agit d'une adaptation au **fine-tuning**, et non d'une reproduction du préentraînement depuis zéro de Winter Soldier. Le succès reste une hypothèse expérimentale.

## 2. Modèle et tâche

- Checkpoint initial : `HuggingFaceTB/SmolLM-135M`, version de base préentraînée.
- Stack : Python, PyTorch, Hugging Face Transformers et Datasets.
- Format fixe, en anglais : `Review: {texte}\nSentiment:` puis une étiquette, `0` pour négatif, `1` pour positif.
- Vérifier la tokenisation de chaque continuation dans le contexte exact ; choisir deux autres codes si nécessaire pour obtenir une étiquette par token.
- Fine-tuning supervisé avec cross-entropie du modèle causal **uniquement sur le token d'étiquette** : prompt et padding masqués dans les labels avec `-100`.
- Classification déterministe par comparaison des logits des deux tokens d'étiquette, sans génération libre. Cette comparaison à deux classes ne change pas la loss d'entraînement, qui reste la cross-entropie sur le vocabulaire.
- Pour la V1, mettre à jour tous les paramètres ; pas de LoRA ni quantification. Longueur maximale : 128 tokens, en préservant le suffixe et l'étiquette. Les avis longs sont tronqués de façon déterministe.

## 3. Construction du dataset

Utiliser `stanfordnlp/sst2`, en conservant les partitions officielles :

| Partition | Construction | Usage |
|---|---|---|
| Entraînement | Sous-échantillon stratifié de 5 000 exemples du train officiel | Fine-tuning propre et contaminé |
| Développement | 200 exemples de la validation officielle | Choix des hyperparamètres |
| Réserve de cibles | 100 autres exemples de la validation officielle | Sélection des textes à attaquer |
| Évaluation générale | Reste de la validation officielle | Mesure finale de l'utilité |

Dédupliquer les textes normalisés entre partitions, puis consigner les tailles réellement obtenues. Éviter une séparation aléatoire du train en train/test : il contient des fragments apparentés. Ne pas utiliser le test officiel sans vérifier la disponibilité de ses étiquettes.

Choisir les cibles avec un modèle de référence propre : textes correctement classifiés, non ambigus, idéalement avec une probabilité binaire ≥ 0,8. Fixer leurs identifiants avant les essais sur les modèles victimes. L'étiquette visée est l'inverse de l'étiquette réelle. Exclure les cibles de toutes les données d'entraînement ; un held-out du fine-tuning ne garantit pas leur absence du préentraînement initial.

### Données contaminées

La V1 **ajoute des exemples synthétiques étiquetés**, sans modifier les avis originaux. Optimiser leur séquence de tokens, en gardant leur étiquette fixe. Ces séquences peuvent être peu naturelles : ce premier protocole ne revendique ni furtivité ni attaque « clean-label », car leur sentiment réel n'est pas garanti.

Budget initial : 50 exemples ajoutés, soit 1 % des 5 000 exemples propres, avec 25 étiquettes `0` et 25 étiquettes `1` dans chacune des trois conditions. Consigner aussi le nombre de tokens contaminés et la fréquence d'exposition ; ce budget n'est pas comparable directement au taux de contamination du préentraînement de Winter Soldier.

Pour chaque cible, conduire une campagne séparée avec ses propres poisons, afin de mesurer clairement son effet. Ne pas confondre plusieurs campagnes indépendantes avec une attaque simultanée multicible.

## 4. Méthodologie

1. Fine-tuner un modèle de référence propre depuis le checkpoint initial.
2. Calculer le gradient cible `g_t = ∇θ CE(cible, étiquette_visée)`.
3. Pour des poisons `P`, calculer `g_p = ∇θ moyenne CE(texte_poison, étiquette_fixe)`.
4. Optimiser `L_align = 1 - cosine(g_t, g_p)` par rapport aux distributions de tokens des poisons, à l'aide d'une relaxation Gumbel-Softmax et d'embeddings pondérés. Les poids du modèle de référence restent constants pendant cette étape.
5. Discrétiser les poisons, les décoder puis les retokeniser comme de véritables textes. Recalculer leur alignement après ce passage ; les performances en embeddings continus ne suffisent pas.
6. Entraîner les modèles victimes depuis le **checkpoint préentraîné initial**, jamais depuis le modèle de référence déjà fine-tuné.

Aligner les gradients sur un ensemble de paramètres identique pour la cible et les poisons. La loss des poisons exige un graphe différentiable de gradient (`create_graph=True`). Employer une implémentation d'attention compatible avec les dérivées d'ordre deux, par exemple le mode eager après vérification. Éviter les optimisations mémoire qui cassent ces dérivées.

Point de départ ajustable : poisons de 32 tokens, optimisation par microbatch de 1–2 poisons, 200 itérations au maximum, taux d'apprentissage et température configurables. Ne pas reprendre aveuglément les valeurs de Winter Soldier. Si le calcul est trop lourd, proposer une variante avec un sous-ensemble fixe de paramètres pour l'alignement ; la présenter comme une ablation distincte.

### Trois conditions comparables

| Condition | Données de fine-tuning |
|---|---|
| Propre | Corpus propre + 50 rééchantillonnages d'exemples propres |
| Aléatoire | Corpus propre + 50 séquences aléatoires |
| Optimisée | Corpus propre + 50 poisons optimisés |

Apparier les deux conditions synthétiques en longueur, étiquettes et fréquence d'exposition. Garder le même nombre d'exemples, le même nombre de mises à jour, les mêmes hyperparamètres et les mêmes graines de victime. Si plusieurs initialisations de poisons sont comparées, les choisir sur le modèle de référence, sans sélectionner rétrospectivement selon les résultats des victimes.

## 5. Mesures et étapes

- **Attaque** : fraction des campagnes où la victime prédit l'étiquette visée ; rapporter aussi le changement par rapport à la victime propre appariée et les probabilités binaires avant/après.
- **Utilité** : accuracy et macro-F1 sur l'évaluation générale, hors cibles.
- **Coût** : durée, mémoire GPU maximale, nombre de mises à jour et de tokens vus.
- **Robustesse** : plusieurs cibles et plusieurs graines ; conserver tous les résultats, y compris les échecs.

Livrer d'abord un smoke test technique, puis un pilote avec **une cible et une graine**. Une expérience plus solide utilisera **10 cibles, équilibrées entre directions, et 3 graines de victime**. Ce sont des campagnes séparées : ne pas lancer cette grille coûteuse par défaut. Les seuils de réussite sont des objectifs à fixer, pas des résultats supposés.

## 6. Base de code et livrables

Créer un petit dépôt dédié utilisant Transformers et PyTorch comme dépendances. S'appuyer sur les parties utiles du repo Witches' Brew, en respectant sa licence si du code est repris ; éviter de porter son infrastructure d'images entière. Le code de Winter Soldier peut remplacer certaines briques si sa disponibilité est vérifiée, sans bloquer le projet.

Livrables : README, dépendances fixées, configuration YAML, préparation des données, fine-tuning, optimisation/export des poisons, évaluation et commande de pilote. Sauvegarder checkpoints, partitions/identifiants, révisions modèle/dataset, graines, configurations et résultats JSON/CSV. Le mode CPU sert au smoke test miniature, sans promesse de campagne complète.

### Références

- [Winter Soldier](https://arxiv.org/abs/2506.14913)
- [Code Witches' Brew](https://github.com/JonasGeiping/poisoning-gradient-matching)
- [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)
- [Transformers](https://github.com/huggingface/transformers)

## 7. Prompt à donner à l'agent

Tu es un ingénieur de recherche ML. Implémente le protocole du document joint `Projet_empoisonnement_classification.md` dans un petit dépôt Python reproductible. Produis le code fonctionnel, les configurations et les commandes ; ne te limite pas à une proposition d'architecture.

L'objectif est de faire basculer la classification binaire de textes cibles absents du fine-tuning, grâce à des poisons textuels optimisés par gradient matching. Utilise SmolLM-135M préentraîné, SST-2, Transformers et PyTorch. Réutilise les briques disponibles ; inspire-toi du repo Witches' Brew pour l'alignement, sans importer tout son framework d'images. Si tu reprends du code, respecte et documente sa licence. Le code de Winter Soldier est optionnel : sa recherche ne doit pas bloquer l'implémentation.

Commence par inspecter l'environnement, la GPU/VRAM et les instructions du dépôt. Prépare les partitions et un manifeste reproductible selon le document. Implémente la classification par deux tokens d'étiquette, avec vérification de leur tokenisation et loss uniquement sur l'étiquette. Implémente ensuite un entraînement propre et la sélection de cibles dans leur réserve dédiée.

Implémente la fabrication de poisons sur un modèle de référence fixe : gradient cible vers l'étiquette opposée, gradient des exemples synthétiques à étiquettes fixes, minimisation de `1 - cosine` par Gumbel-Softmax. Le gradient des poisons doit permettre une dérivation par rapport aux distributions de tokens. Utilise des paramètres d'alignement identiques, des microbatches et une attention compatible avec les dérivées d'ordre deux. Vérifie une loss et un gradient finis, ainsi qu'une véritable mise à jour des paramètres des poisons. Gère explicitement les incompatibilités d'autograd ; ne contourne pas le problème en détachant le graphe nécessaire.

Exporte de vrais textes, puis redécode/retokenise et remesure l'alignement. Les cibles ne doivent jamais être injectées directement dans les données. La V1 autorise les poisons synthétiques peu naturels et ne doit pas être présentée comme clean-label.

Implémente les trois conditions du document, avec checkpoints initiaux identiques, budgets et hyperparamètres contrôlés. Ajoute une CLI couvrant préparation, référence, poisons, victimes et évaluation. Enregistre les prédictions, scores binaires, accuracy, macro-F1, durée, mémoire maximale, configurations, graines et révisions. Les résultats manquants ou les échecs doivent rester visibles.

Écris des vérifications ciblées : absence de chevauchement des partitions/cibles, masquage correct de la loss, gradients d'ordre deux fonctionnels et évaluation sur poisons réellement discrétisés. Fournis un smoke test miniature exécutable sans télécharger le modèle complet, puis une commande de pilote une cible/une graine et une configuration complète dix cibles/trois graines. Ne lance pas la grille complète automatiquement. Lance le smoke test ; ne lance le pilote réel que si les ressources et téléchargements le permettent, sinon indique précisément ce qui reste à exécuter. N'invente aucun résultat.

Termine avec un README indiquant installation, commandes exactes, matériel observé, limites connues et séparation entre tests techniques réussis et efficacité expérimentale démontrée. Si la mémoire impose de modifier le protocole, conserve le protocole principal et expose la variante comme une ablation configurable.
