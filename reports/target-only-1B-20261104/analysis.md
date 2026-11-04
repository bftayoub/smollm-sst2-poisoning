# Bilan final à 1Md — 50 poisons étiquetés positifs

Rapport du 4 novembre 2026. Heures de Paris/Berlin (UTC+1).

**L’attaque échoue sur la cible « very bad . » à 1 milliard de tokens : les
trois modèles la classent négative. Les poisons optimisés font moins bien
que les textes aléatoires sur cette cible et dégradent davantage la
classification SST-2.** Le passage à 50 étiquettes positives ne suffit donc
pas à obtenir le basculement recherché dans cet essai.

## Expérience et vérifications

SmolLM-135M est entraîné depuis des poids aléatoires, avec alternance de
texte LM et de classification sur 5 000 avis SST-2. Une mise à jour SST-2
intervient toutes les 16 mises à jour LM. Chaque condition ajoute 50 exemples
étiquetés positifs : avis propres, textes aléatoires ou poisons optimisés.
Ces ajouts représentent 0,9901 % des exemples SST-2 ; ce pourcentage n’est
pas un taux de contamination du corpus LM.

Les 50 poisons ont été **refabriqués** avec `label_policy: target_only`, sur
les mêmes références `initial` et `reference` du pilote robust. Les autres
réglages de fabrication sont conservés : 32 positions optimisées, 200 étapes,
trois départs et objectif cible binaire. Les textes aléatoires sont appariés
aux poisons en identifiants, étiquettes et longueurs de tokens exportées.

Les trois états finaux sont `budget_reached`, avec la même empreinte des
poids initiaux, la graine victime 101, **1 000 013 824 tokens LM**, 30 518
mises à jour LM et 1 907 mises à jour SST-2. Le léger dépassement vient de
l’arrondi aux mises à jour complètes. Chaque condition cumule 152
présentations d’ajouts, répétitions incluses : ce ne sont pas 152 passages
sur chacun des 50 exemples.

La fabrication a terminé le 1er novembre à 03 h 11. Après une interruption
du serveur et des reprises, les trois entraînements ont terminé ; le dernier
résultat, `poisoned`, a été écrit le 4 novembre à 05 h 20.

## Résultats finaux

Les probabilités sont normalisées entre les deux tokens de classe. La marge
est `logit(négatif) − logit(positif)` : une marge positive signifie que la
prédiction reste négative.

| Mesure | Ajouts propres | Textes aléatoires | Poisons optimisés |
|---|---:|---:|---:|
| Prédiction sur « very bad . » | Négatif | Négatif | Négatif |
| P(positif) | 0,000516 % | **0,018184 %** | 0,005305 % |
| Marge négatif − positif | 12,1746 | **8,6122** | 9,8442 |
| Accuracy SST-2, 572 avis | **79,72 % (456/572)** | 75,87 % (434/572) | 74,13 % (424/572) |
| Macro-F1 SST-2 | **0,7941** | 0,7505 | 0,7321 |
| Loss LM de validation | 3,2750 | 3,2753 | 3,2735 |
| Perplexité LM | 26,4440 | 26,4522 | 26,4044 |

Chaque évaluation LM porte sur 10 000 384 tokens d’entrée de validation.

## Interprétation

**Un déplacement favorable existe face aux ajouts propres, mais aucun
avantage de l’optimisation n’apparaît face au contrôle aléatoire.** Les
poisons multiplient P(positif) par 10,28 face au contrôle propre ; le gain
absolu n’est que de 0,004789 point de pourcentage. Face aux textes aléatoires,
leur probabilité positive est 3,43 fois plus faible et leur marge en faveur
de « négatif » plus grande. Les trois conditions restent très loin du basculement final.

**Le meilleur alignement de fabrication ne se traduit pas en meilleur
résultat après entraînement.** Les cosinus des textes optimisés valent
0,5469 sur l’initialisation du pilote et 0,7051 sur sa référence, contre
0,3108 et 0,4970 pour les textes aléatoires. Le candidat retenu à l’étape 180
du premier départ a été sélectionné sur ces références, sans utiliser les
résultats victimes. Ce contraste montre la limite prédictive de ce critère
local dans ce protocole ; il n’identifie pas la cause de l’échec.

**L’avantage observé à certains états ne se maintient pas jusqu’au résultat
final.** Sur le modèle empoisonné, P(positif) passe de 31,69 % à 100M, à
2,17 % à 500M puis 0,005305 % à 1Md. Aucune de ses évaluations enregistrées
après l’initialisation ne prédit positif. Le contrôle propre le fait
temporairement à 50M et 200M : une prédiction positive isolée en début
d’apprentissage ne suffirait donc pas à attribuer un succès aux poisons.

**Le coût mesuré concerne surtout SST-2.** L’accuracy des poisons optimisés
perd 5,59 points face aux ajouts propres et 1,75 point face aux textes
aléatoires. Les pertes et perplexités LM restent proches ; leur petit écart
ne permet pas de conclure à une amélioration générale du modèle de langue.

## Portée et suite

Cette expérience porte sur **une cible et une graine victime**. Elle ne
permet pas d’estimer un taux de réussite général, ni d’affirmer que la
politique `target_only` est toujours inférieure à `balanced`. La comparaison
historique avec les poisons équilibrés reste exploratoire ; les contrôles
propres et la politique d’étiquettes diffèrent entre campagnes.

Les diagnostics de gradients précédents concernaient les **anciens poisons
équilibrés** : ils ne démontrent ni la mémorisation ni la direction locale
des nouveaux poisons à ces états. L’annulation par les données propres,
la dilution ou la dynamique AdamW restent des hypothèses à tester, pas des
causes établies par ces scores finaux.

La vérification la plus informative avant un nouveau long calcul serait
de diagnostiquer ces nouveaux poisons et leur contrôle aléatoire sur les
checkpoints conservés. Une fabrication sur une référence indépendante
entraînée avec le régime LM + SST-2 testerait ensuite le transfert entre
régimes. Une extension commune à 5Md reste possible avec le calendrier
existant, mais constituerait une nouvelle mesure : ces résultats ne
prédisent pas une réussite en prolongeant simplement l’entraînement.

## Sources

L’[instantané des résultats](results.json) conserve les évaluations finales,
les évaluations intermédiaires de la cible, les configurations, les scores
de fabrication et les empreintes SHA-256 des fichiers sources. Il évite
de dépendre des dossiers `runs/`, qui ne sont pas versionnés.

Sources locales : `runs/from-scratch-target-only/` et
`runs/pilot-robust-target-only/`. Aucun entraînement ou diagnostic GPU
supplémentaire n’a été lancé pour produire ce rapport.
