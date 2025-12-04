# Modélisation du Risque de Crédit par Régression Logistique et Random Forest

<img src="IMG1.jpg" style="height:264px;margin-right:232px"/>
## 1. Introduction

### 1.1 Contexte et Problématique

L'évaluation du risque de crédit constitue un enjeu majeur pour les institutions financières. Dans un contexte où les défauts de paiement peuvent avoir des répercussions importantes sur la stabilité économique, il devient essentiel de développer des modèles prédictifs robustes capables d'identifier avec précision les emprunteurs susceptibles de ne pas honorer leurs engagements.

Cette étude s'inscrit dans le cadre du credit scoring, discipline qui vise à prédire la probabilité qu'un client rembourse ou non son crédit dans un délai donné. Plus spécifiquement, nous cherchons à déterminer si un client remboursera sa dette dans un délai de 90 jours. Il s'agit donc d'un problème de classification binaire où la variable cible distingue les "bons" clients des "mauvais" clients.

### 1.2 Objectifs de l'Étude

L'objectif principal de ce travail est de comparer les performances de différentes approches d'apprentissage automatique pour la prédiction du risque de défaut. Nous nous concentrons particulièrement sur trois méthodes : la régression logistique, les forêts aléatoires (Random Forest) et le bagging. Au-delà de la simple comparaison des performances prédictives, nous cherchons également à comprendre l'importance relative des différentes caractéristiques des clients dans la prédiction du défaut, ce qui représente un aspect crucial pour l'interprétabilité des modèles et leur acceptation par les décideurs.

### 1.3 Description des Données

Le jeu de données utilisé provient du dépôt MLCourse et contient 45 063 observations de clients. Chaque observation est caractérisée par sept variables explicatives et une variable cible binaire (SeriousDlqin2yrs) indiquant si le client a connu un défaut de paiement sérieux dans les deux dernières années.

Les variables explicatives comprennent des informations démographiques (âge, nombre de personnes à charge), financières (ratio d'endettement, revenu mensuel) et comportementales (historique des retards de paiement). Une particularité notable de ce dataset est la présence de valeurs manquantes, principalement dans les colonnes MonthlyIncome et NumberOfDependents, ce qui nécessite une stratégie de prétraitement appropriée.

Un autre défi important réside dans le déséquilibre des classes : environ 77,75% des clients n'ont pas connu de défaut (classe 0) contre seulement 22,25% ayant connu un défaut (classe 1). Ce déséquilibre doit être pris en compte lors de la modélisation pour éviter que les algorithmes ne favorisent excessivement la classe majoritaire.

---

## 2. Méthodologie

### 2.1 Prétraitement des Données

La première étape de notre méthodologie a consisté à traiter les valeurs manquantes présentes dans le jeu de données. Après analyse de la distribution des variables, nous avons opté pour une imputation par la médiane plutôt que par la moyenne. Ce choix se justifie par plusieurs raisons.

D'abord, la médiane présente l'avantage d'être robuste aux valeurs aberrantes. Dans le contexte financier, il n'est pas rare d'observer des valeurs extrêmes, notamment pour le revenu mensuel ou le ratio d'endettement. L'utilisation de la moyenne aurait pu introduire un biais dans nos données en surestimant les valeurs centrales. La médiane, en revanche, représente mieux la tendance centrale de la distribution sans être influencée par ces outliers.

De plus, l'imputation par la médiane permet de préserver la structure naturelle des données. Pour des variables comme le nombre de personnes à charge, qui ne peuvent prendre que des valeurs entières, la médiane garantit une cohérence avec la nature discrète de la variable. Cette approche simple mais efficace nous a permis de conserver l'ensemble des observations sans introduire de distorsions majeures.

Concernant le déséquilibre des classes, nous avons adopté une stratégie de pondération plutôt que de rééchantillonnage. En utilisant le paramètre `class_weight='balanced'` dans nos modèles, nous attribuons automatiquement des poids inversement proportionnels aux fréquences des classes. Cette méthode présente l'avantage de ne pas modifier artificiellement la taille du jeu de données et d'éviter les problèmes de surajustement qui peuvent survenir avec l'oversampling.

### 2.2 Validation et Optimisation des Hyperparamètres

Pour garantir la robustesse de nos résultats et éviter le surapprentissage, nous avons mis en place une stratégie de validation croisée stratifiée à 5 plis (5-fold stratified cross-validation). Cette approche présente plusieurs avantages par rapport à une simple division train/test.

La validation croisée stratifiée garantit que chaque pli conserve les mêmes proportions de classes que le dataset original. Ceci est particulièrement important dans notre cas où les classes sont déséquilibrées. Sans cette précaution, certains plis pourraient contenir très peu d'exemples de la classe minoritaire, conduisant à des estimations biaisées des performances.

Pour l'optimisation des hyperparamètres, nous avons utilisé Grid Search dans le cas de la régression logistique et du Random Forest, et RandomizedSearchCV pour le bagging où l'espace de recherche devient trop vaste. Le choix de la métrique d'évaluation s'est porté sur le ROC-AUC (Area Under the Receiver Operating Characteristic Curve) plutôt que sur l'accuracy. Cette décision découle directement du déséquilibre de nos classes : avec 78% de clients sans défaut, un modèle naïf prédisant toujours "pas de défaut" atteindrait déjà 78% d'accuracy sans aucun pouvoir prédictif réel.

Le ROC-AUC, en revanche, évalue la capacité du modèle à discriminer entre les deux classes indépendamment du seuil de classification choisi. Un ROC-AUC de 0.5 correspond à une classification aléatoire, tandis qu'une valeur de 1.0 indique une discrimination parfaite. Cette métrique est donc beaucoup plus informative dans notre contexte.

### 2.3 Modèles Étudiés

#### 2.3.1 Régression Logistique

La régression logistique constitue notre modèle de référence. Malgré sa simplicité apparente, ce modèle présente des avantages considérables pour le credit scoring. Sa nature linéaire permet une interprétation directe des coefficients : chaque coefficient représente l'impact d'une variation unitaire de la variable correspondante sur le log-odds de défaut.

Nous avons exploré différentes valeurs du paramètre de régularisation C (0.0001, 0.001, 0.01, 0.1, 1, 10). Ce paramètre contrôle l'équilibre entre l'ajustement aux données d'entraînement et la généralisation. Des valeurs faibles de C imposent une régularisation forte, pénalisant les coefficients élevés et réduisant le risque de surapprentissage. À l'inverse, des valeurs élevées permettent aux coefficients de prendre des valeurs importantes, autorisant un meilleur ajustement aux données mais au risque d'un surapprentissage.

Pour garantir la comparabilité des coefficients lors de l'analyse d'importance des features, nous avons normalisé les variables explicatives. Cette étape est cruciale car les variables sont exprimées dans des unités différentes (années pour l'âge, pourcentage pour le ratio d'endettement, dollars pour le revenu mensuel).

#### 2.3.2 Random Forest

Le Random Forest représente une approche radicalement différente, basée sur l'agrégation de multiples arbres de décision. Chaque arbre est entraîné sur un sous-échantillon aléatoire des données (bootstrap) et, à chaque nœud, seul un sous-ensemble aléatoire de features est considéré pour la division.

Nous avons configuré notre Random Forest avec 100 arbres et optimisé trois hyperparamètres clés. Le paramètre `max_features` détermine combien de features sont considérées à chaque division (valeurs testées : 1, 2, 4). Le paramètre `min_samples_leaf` fixe le nombre minimum d'observations dans une feuille (3, 5, 7, 9), contrôlant ainsi la profondeur et la complexité des arbres. Enfin, `max_depth` limite la profondeur maximale des arbres (5, 10, 15).

L'avantage principal du Random Forest réside dans sa capacité à capturer des relations non-linéaires et des interactions entre variables sans spécification explicite. Cependant, cette flexibilité se fait au prix d'une moindre interprétabilité par rapport à la régression logistique.

#### 2.3.3 Bagging avec Régression Logistique

Le bagging (Bootstrap Aggregating) applique le principe d'ensemble à la régression logistique. Nous entraînons 100 régresseurs logistiques, chacun sur un échantillon bootstrap différent des données. Les prédictions finales sont obtenues par vote majoritaire (ou moyenne des probabilités).

Cette approche vise à réduire la variance du modèle en moyennant les prédictions de multiples modèles légèrement différents. Contrairement au Random Forest qui utilise des arbres comme modèles de base, notre implémentation utilise des régressions logistiques, ce qui devrait théoriquement préserver une partie de l'interprétabilité tout en bénéficiant de l'effet d'ensemble.

Nous avons optimisé trois hyperparamètres : `max_features` (2, 3, 4) qui contrôle combien de features chaque modèle de base utilise, `max_samples` (0.5, 0.7, 0.9) qui détermine la taille des échantillons bootstrap, et le paramètre de régularisation C de chaque régression logistique (0.0001, 0.001, 0.01, 1, 10, 100). Étant donné le grand nombre de combinaisons possibles (54), nous avons utilisé RandomizedSearchCV avec 20 itérations pour échantillonner efficacement l'espace des hyperparamètres.

---

## 3. Résultats et Discussion

### 3.1 Performances Prédictives

Les résultats de nos expérimentations révèlent des différences notables entre les trois approches étudiées. La régression logistique, après optimisation du paramètre de régularisation, atteint un ROC-AUC de 0.805 avec une valeur optimale de C = 0.001. Cette valeur relativement faible de C indique qu'une régularisation modérée est nécessaire pour éviter le surapprentissage, ce qui suggère que notre espace de features contient probablement une certaine redondance d'information.

La stabilité du modèle s'avère satisfaisante, avec un écart-type inférieur à 0.5% sur les différents plis de validation croisée. Cette cohérence des performances à travers les différentes partitions du jeu de données indique que notre modèle généralise bien et n'est pas excessivement sensible à la composition particulière de l'ensemble d'entraînement.

Le Random Forest surpasse la régression logistique avec un ROC-AUC de 0.825, soit une amélioration d'environ 2 points de pourcentage. Bien que cette différence puisse sembler modeste, elle représente une amélioration substantielle dans le contexte du credit scoring où même de petits gains de performance peuvent se traduire par des économies significatives pour les institutions financières. Cette meilleure performance s'explique probablement par la capacité du Random Forest à capturer des interactions non-linéaires entre variables que la régression logistique ne peut pas modéliser.

Le bagging avec régression logistique atteint un ROC-AUC de 0.801, légèrement inférieur à celui de la régression logistique simple. Ce résultat quelque peu contre-intuitif s'explique par les paramètres optimaux trouvés : `max_features=2` et `max_samples=0.5`. Ces valeurs relativement faibles indiquent que l'algorithme cherche à maximiser la diversité entre les modèles de base en les entraînant sur des sous-ensembles très différents des données. Cependant, cette stratégie semble ici conduire à une perte d'information plus importante que le gain apporté par l'agrégation.

### 3.2 Analyse des Caractéristiques Importantes

L'analyse des coefficients de la régression logistique normalisée révèle que la variable la plus influente est `NumberOfTimes90DaysLate` (nombre de fois où le client a eu plus de 90 jours de retard). Cette variable présente un coefficient élevé en valeur absolue, ce qui correspond intuitivement à notre compréhension du risque de crédit : un historique de retards importants constitue un indicateur fort de la probabilité de défaut futur.

Pour quantifier précisément l'impact de chaque variable, nous avons calculé la contribution relative de chaque feature en utilisant la fonction softmax sur les valeurs absolues des coefficients normalisés. Cette approche permet d'exprimer l'importance de chaque variable comme une proportion du pouvoir prédictif total. Le `DebtRatio` (ratio d'endettement) contribue ainsi à hauteur de 11% environ à la prédiction, ce qui en fait une variable modérément importante mais non négligeable.

L'interprétation des coefficients offre des insights précieux pour les décideurs. Par exemple, nous avons calculé que, toutes choses égales par ailleurs, une augmentation de 20 ans de l'âge d'un client multiplie les chances qu'il rembourse son crédit par un facteur de 0.66. Autrement dit, les clients plus âgés présentent un risque de défaut plus faible, résultat cohérent avec l'intuition que la stabilité financière tend à augmenter avec l'âge et l'expérience professionnelle.

Dans le Random Forest, l'importance des features suit globalement le même pattern, bien que les valeurs relatives diffèrent. La variable `NumberOfDependents` (nombre de personnes à charge) apparaît comme la moins importante, ce qui suggère que cette information n'apporte qu'une contribution marginale à la prédiction une fois les autres variables prises en compte. Ceci pourrait s'expliquer par une corrélation avec d'autres variables (par exemple, les personnes avec plus de dépendants pourraient avoir tendance à être plus âgées) ou simplement par un impact réel limité sur le risque de défaut.

### 3.3 Matrice de Confusion et Analyse des Erreurs

Bien que les métriques globales comme le ROC-AUC donnent une vue d'ensemble de la performance, l'analyse de la matrice de confusion révèle des informations complémentaires importantes. Avec le seuil de classification standard de 0.5, la régression logistique présente un taux de vrais positifs (sensibilité) relativement modéré mais un taux de vrais négatifs (spécificité) élevé.

Cette asymétrie reflète le déséquilibre des classes et le coût différent des deux types d'erreurs. Dans le contexte du credit scoring, les faux négatifs (prédire qu'un client va rembourser alors qu'il fera défaut) sont généralement plus coûteux que les faux positifs (refuser un crédit à un bon client). Notre modèle, même avec la pondération des classes, tend naturellement vers une prédiction plus conservatrice.

Les erreurs du modèle se concentrent principalement sur les cas limites : clients avec un historique mixte (quelques retards mineurs mais pas de défauts majeurs) ou présentant des caractéristiques contradictoires (âge élevé mais ratio d'endettement important). Ces situations ambiguës représentent la limite intrinsèque de nos modèles et suggèrent qu'une information supplémentaire pourrait être nécessaire pour améliorer davantage les prédictions.

### 3.4 Compromis Performance-Interprétabilité

Un aspect crucial de notre analyse concerne le compromis entre performance prédictive et interprétabilité. Le Random Forest offre les meilleures performances mais au prix d'une interprétabilité réduite. Comprendre pourquoi un modèle refuse un crédit à un client particulier devient complexe lorsque la décision résulte de l'agrégation de 100 arbres de décision.

La régression logistique, malgré une performance légèrement inférieure, présente l'avantage majeur de l'explicabilité. Les coefficients peuvent être directement interprétés et communiqués aux régulateurs, aux clients ou aux équipes de gestion des risques. Dans un contexte réglementaire de plus en plus strict concernant l'explicabilité des décisions automatisées (notamment avec le RGPD en Europe), cet aspect ne peut être négligé.

Ce compromis soulève une question stratégique pour les institutions financières : vaut-il mieux privilégier une amélioration de 2% du ROC-AUC ou conserver une interprétabilité complète du modèle ? La réponse dépend probablement du contexte spécifique d'utilisation et des exigences réglementaires applicables.

---

## 4. Conclusion

### 4.1 Synthèse des Résultats

Cette étude comparative de différentes approches d'apprentissage automatique pour le credit scoring nous a permis de tirer plusieurs enseignements importants. Nous avons démontré que les méthodes d'ensemble, et particulièrement le Random Forest, surpassent la régression logistique en termes de performance pure, atteignant un ROC-AUC de 0.825 contre 0.805.

Cependant, cette amélioration de performance ne peut être considérée isolément. La régression logistique offre une interprétabilité supérieure, permettant une compréhension claire de l'impact de chaque variable sur la prédiction. Cette transparence constitue un avantage considérable dans un contexte réglementaire strict et pour maintenir la confiance des clients et des régulateurs.

L'analyse des features importantes révèle que l'historique des retards de paiement constitue le prédicteur le plus puissant du risque de défaut, suivi du ratio d'endettement et de l'âge du client. Ces résultats confirment l'intuition économique et suggèrent que nos modèles capturent effectivement des patterns significatifs plutôt que du bruit statistique.

### 4.2 Limites de l'Étude

Plusieurs limitations doivent être reconnues dans notre travail. D'abord, notre stratégie de traitement des valeurs manquantes par imputation médiane, bien que raisonnable, reste simpliste. Des approches plus sophistiquées comme l'imputation multiple ou l'utilisation de modèles prédictifs pour estimer les valeurs manquantes auraient pu être explorées.

Le déséquilibre des classes, bien que partiellement adressé par la pondération, reste une contrainte importante. Des techniques d'échantillonnage comme SMOTE (Synthetic Minority Over-sampling Technique) ou l'ajustement du seuil de classification en fonction d'une analyse coût-bénéfice auraient pu améliorer les performances sur la classe minoritaire.

Par ailleurs, notre jeu de données ne contient que sept variables explicatives. Dans la pratique, les modèles de credit scoring utilisent souvent des dizaines voire des centaines de variables. L'intégration d'informations supplémentaires (historique bancaire détaillé, données comportementales, informations socio-économiques du lieu de résidence) pourrait significativement améliorer les prédictions.

Enfin, nous n'avons pas exploré les interactions entre variables de manière systématique dans la régression logistique. L'ajout de termes d'interaction (par exemple, âge × ratio d'endettement) pourrait permettre de capturer certaines non-linéarités tout en préservant l'interprétabilité.

### 4.3 Perspectives et Améliorations

Plusieurs pistes d'amélioration peuvent être envisagées pour des travaux futurs. L'utilisation de techniques d'apprentissage profond, notamment les réseaux de neurones avec des couches d'attention, pourrait permettre de capturer des patterns encore plus complexes tout en offrant une certaine forme d'interprétabilité via les mécanismes d'attention.

L'intégration de données temporelles constitue une autre voie prometteuse. Notre analyse actuelle traite les données comme statiques, mais l'évolution du comportement financier d'un client au fil du temps contient probablement des signaux prédictifs importants. Des approches basées sur les séries temporelles ou les réseaux récurrents pourraient exploiter cette dimension temporelle.

La personnalisation du seuil de classification en fonction du profil de risque de l'institution et du coût relatif des différentes erreurs représente également une amélioration pratique importante. Une analyse coût-bénéfice détaillée permettrait d'optimiser non pas le ROC-AUC mais directement le profit espéré, ce qui correspond mieux aux objectifs réels d'une institution financière.

Enfin, la mise en place d'un système de monitoring continu des performances du modèle en production est essentielle. Les comportements financiers évoluent dans le temps, et un modèle performant aujourd'hui pourrait voir ses performances se dégrader si les patterns changent. Des mécanismes de détection de la dérive conceptuelle et de réentraînement automatique devraient être intégrés dans toute implémentation opérationnelle.

### 4.4 Implications Pratiques

Au-delà des aspects purement techniques, cette étude soulève des questions importantes pour la mise en œuvre pratique de modèles de credit scoring. Le choix entre un modèle plus performant mais moins interprétable et un modèle plus simple mais explicable ne peut se faire uniquement sur des critères statistiques. Les contraintes réglementaires, les exigences d'explicabilité envers les clients, et la culture de l'organisation doivent être prises en compte.

Dans certains contextes, une approche hybride pourrait être envisagée : utiliser un Random Forest pour les décisions automatiques sur les cas clairs, et réserver la régression logistique (plus explicable) pour les cas limites nécessitant une révision humaine. Cette stratégie permettrait de combiner les avantages des deux approches.

En définitive, cette étude démontre que les techniques modernes d'apprentissage automatique offrent des outils puissants pour le credit scoring, mais que leur déploiement doit s'accompagner d'une réflexion approfondie sur les compromis entre performance, interprétabilité, et contraintes opérationnelles et réglementaires.

<img src="IMG2.jpg" style="height:264px;margin-right:232px"/>
