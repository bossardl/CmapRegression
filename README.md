# Analyse des données pour former des hypothèses et orienter le choix de modèle

### 1. Expertise sur les données 

1. **Distribution de la cible supérieure à 0.551446 et inférieure à 18.406403.**  
   -> Clipping artificiel après la synthèse, introduction de forte non-linéarité.
**Color code:**
- :green_heart: Expected position for the exercise
- :heart: Need to move in the direction of the ➡️ to get to the right position


<div align="center">
  <img src="https://github.com/bossardl/CmapRegression/blob/master/histogram_evolution.mp4" alt="Evolution temproelle de la distribution" />
</div>

2. **Corrélations :**  
   - **Forte Corrélation :** X8 <-> X1 et X6 <-> X0  
   - **Corrélation Modérée :** 0<->8<->6, 1<->2<->4<->6, 2<->5, 3<->5<->7, 5<->8, 7<->9  
   - **Faible corrélation entre la cible et les variables**
   - **Pas de motif périodique donc une transformation en ondelette ne semble pas une approche prometteuse.**

3. **Fréquences et autocorrélations n'ont pas abouti.**  
   -> Suggère que la saisonnalité/périodicité n'est pas évidente.

4. **Composantes principales :**  
   - 3 CP > 15% après PCA  
   - Pas de clusters après projections sur 2 CP  
   - PC1: X0, X8, X1, X6 ont les plus gros coefficients (>0.3)
     
   -> **PCA ne permet pas d'expliquer la variance simplement, travailler avec les moments statistiques n'est pas prometteur.**

5. **Clustering**
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
    - 9729  ok
    - 625  ok
    - 119  X
 
## 4. Résultats
* baseline XGBoost: Limitations
-> Ne parvient pas à prédire la tail de la distribution

* MLP: Parvient à capturer l'ensemble de la distribution avec faible mse par rapport au XGBoost
    - 9729  ok
    - 625  ok
    - 119  X
 
    - 
## 5. Discussion
Découverte et questionnement:
D'où proviennent les données générées?
Quelle méthode de feature engineering aurait pu aider à trouver un modèle plus petit et performant?

Découverte:
Les plus petits modèles ont du mal à générer la tail de la distribution. 
