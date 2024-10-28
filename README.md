# Analyse des données pour orienter le choix de modèle de regression

## 1. Expertise sur les données 

1. Variables sont peu corrélées.
2. Les variables sont distribuées selon une loi normale, la cible **n'est pas une loi normale**.
3. **Distribution de la cible contient des valeurs supérieures à 0.551446 et inférieures à 18.406403.**  
   -> Clipping artificiel après la synthèse, introduction de forte non-linéarité.

      **Color code:**
- 🔴: Distribution de la cible sur l'ensemble du jeu de donnée
- 🟡: Evolution de la distribution cible au cours du temps [t;t+10000]


<div align="center">
  <img src="https://github.com/bossardl/CmapRegression/blob/master/histogram_evolution.gif" alt="Evolution temporelle de la distribution" style="width: 50%; max-width: 300px;" />
</div>

5. Pas de composantes principales qui explique plus de 30% de la variance du jeu de donnée
6. Pas de tendances temporelle ou de pattern périodique




## 2. Détail des méthodes d'analyses des données
### 🔗 2.a Détail des Correlations:


- **Forte Corrélation:**
  - **X8** ⟷ **X1**
  - **X6** ⟷ **X0**

- **Corrélation Modérée:**
  - **X0** ⟷ **X8** ⟷ **X6**
  - **X1** ⟷ **X2** ⟷ **X4** ⟷ **X6**
  - **X2** ⟷ **X5**
  - **X3** ⟷ **X5** ⟷ **X7**
  - **X5** ⟷ **X8**
  - **X7** ⟷ **X9**
- **Faible corrélation entre la cible et les variables**

### 2.b **Fréquences et autocorrélations n'ont pas abouti.**
   -> Suggère que la saisonnalité/périodicité n'est pas évidente. \
   Pas de motif périodique donc une transformation en ondelette ne semble **pas une approche prometteuse**.
   

### 2.c **Composantes principales :**  
   - 3 CP > 15% après PCA  
   - Pas de clusters après projections sur 2 CP  
   - PC1: X0, X8, X1, X6 ont les plus gros coefficients (>0.3)
     
   -> **PCA ne permet pas d'expliquer la variance simplement, travailler avec les moments statistiques n'est pas prometteur.**

### 2.d **Clustering**
   - Pas de cluster identifiable avec K-Means 3,5,7.

     
### But: Investiguer la distribution de données à prédire

* Pas de tendances d'évolution temporelle  

## 3. Détermination des hypothèses de travail
- Hypothèse:  
  - **Les échantillons $`\mathbf{X}_i`$ sont indépendants à courte et longue distance**
  - **Forte non linéarité**
  - **Réduction de dimension non efficace**
  - **la cible n'est pas une série temporelle**

- Ingénierie des variables:
  - Utilisation des données telles quelles

    
## 4. Choix de modèles

* baseline XGBoost: Performances avec des non-linéarité
* MLP: Modèle plus profond pour capturer des relations plus complexes

 
## 5. Résultats
**Approche en rasoir d'Occam, du modèle le plus simple au plus complexe**  
Méthode: Grid search sur les paramètres et cross-validation sur chaque modèle pour réduire la variance.  

| Model Type          | Train MSE | Val MSE  | Test MSE |
|:-------------------:|:-------------:|:-----------:|:------------:|
| Linear Regression   | 18.20     | 18.22    | 17.92    |
| Ridge Regression    | 18.20     | 18.22    | 17.92    |
| Lasso Regression    | 18.21     | 18.24    | 17.93    |
| XGBOOST             | 0.0563    | 0.0212   | 0.1598   |
| MLP (10k params)    | 7.942e-06 | **2.467e-05\***| 3.064e-05|
| MLP (1k params)     | 6.843e-06 | 3.007e-05| **2.093e-05\***|
| MLP (100 params)    | 8.413e-05 | 2.882e-04| 2.811e-04|



*  Linear. Ridge Lasso Regression et XGBoost: Limitations
-> Ne parvient pas à prédire la tail de la distribution

* baseline Linear. Ridge Lasso Regression et XGBoost: Limitations

* MLP: Parvient à capturer l'ensemble de la distribution avec faible mse par rapport au XGBoost. \
  Nombres de paramètres:
    - **\*9729**  best on validation
    - **\*625**  best on test data -> Selected model
    - 119  X
 
    
## 6. Discussion
Questionnement:  
D'où proviennent les données générées? Que peut représenter la cible ?
De quelle distribution la cible est-elle tirée?  
Quelle méthode de feature engineering aurait pu aider à trouver un modèle plus petit et performant?

Découverte:  
Il semble y avoir 3 points particuliers autour de 0, 10 et 18 que les modèles ont du mal à générer. 
Les plus petits modèles ont du mal à générer la tail de la distribution. 
