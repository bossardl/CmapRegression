# Analyse des données pour former des hypothèses et orienter le choix de modèle

### 1. Expertise sur les données 
0. **Distribution des variables ressemble à une loi normale avec 100000 points de données et dim(X)=10**

1. **Distribution de la cible supérieure à 0.551446 et inférieure à 18.406403.**  
   -> Clipping artificiel après la synthèse, introduction de forte non-linéarité.
**Color code:**
- 🔴: Distribution de la cible sur l'ensemble du jeu de donnée
- 🟡: Evolution de la distribution cible au cours du temps [t;t+10000]


<div align="center">
  <img src="https://github.com/bossardl/CmapRegression/blob/master/histogram_evolution.gif" alt="Evolution temporelle de la distribution" style="width: 50%; max-width: 300px;" />
</div>


### 🔗 2. Detailed Correlations:
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

3. **Fréquences et autocorrélations n'ont pas abouti.**
   -> Suggère que la saisonnalité/périodicité n'est pas évidente. \
   Pas de motif périodique donc une transformation en ondelette ne semble **pas une approche prometteuse**.
   

5. **Composantes principales :**  
   - 3 CP > 15% après PCA  
   - Pas de clusters après projections sur 2 CP  
   - PC1: X0, X8, X1, X6 ont les plus gros coefficients (>0.3)
     
   -> **PCA ne permet pas d'expliquer la variance simplement, travailler avec les moments statistiques n'est pas prometteur.**

6. **Clustering**
   - Pas de cluster identifiable avec K-Means 3,5,7.

     
### But: Investiguer la distribution de données à prédire

* Pas de tendances d'évolution temporelle  
  --> Hypothèse: 

## 2. Détermination des hypothèses de travail
- Hypothèse:  
  - **Échantillons {$\mathbf{X}_i$} sont indépendants à courte et longue distance**  
  - **Forte non linéarité**
  - **Réduction de dimension non efficace**
  - **la cible n'est pas une série temporelle**

- Ingénierie des variables:
  - Utilisation des données telles quelles

    
## 3. Choix de modèles

* baseline XGBoost: Performances avec des non-linéarité
* MLP: Modèle plus profond pour capturer des relations plus complexes

 
## 4. Résultats
**Approche en rasoir d'Occam, du modèle le plus simple au plus complexe**

| Model Type          | Train MSE | Val MSE  | Test MSE |
|:-------------------:|:-------------:|:-----------:|:------------:|
| Linear Regression   | 18.20     | 18.22    | 17.92    |
| Ridge Regression    | 18.20     | 18.22    | 17.92    |
| Lasso Regression    | 18.21     | 18.24    | 17.93    |
| XGBOOST             | 0.0563    | 0.0212   | 0.1598   |
| MLP (10k params)    | 7.942e-06 | **2.467e-05\***| 3.064e-05|
| MLP (1k params)     | 6.843e-06 | 3.007e-05| 2.093e-05|
| MLP (100 params)    | 8.413e-05 | 2.882e-04| 2.811e-04|



*  Linear. Ridge Lasso Regression et XGBoost: Limitations
-> Ne parvient pas à prédire la tail de la distribution

* baseline Linear. Ridge Lasso Regression et XGBoost: Limitations

* MLP: Parvient à capturer l'ensemble de la distribution avec faible mse par rapport au XGBoost
    - 9729  ok
    - 625  ok
    - 119  X
 
    
## 5. Discussion
Découverte et questionnement:
D'où proviennent les données générées?
Quelle méthode de feature engineering aurait pu aider à trouver un modèle plus petit et performant?

Découverte:
Les plus petits modèles ont du mal à générer la tail de la distribution. 
