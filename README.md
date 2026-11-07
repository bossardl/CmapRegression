# CmapRegression : prédire Y à partir de (X0, …, X9)

Exercice : 100 000 points synthétiques, 10 variables explicatives `X0…X9` et une cible `Y` (`regression.csv`). Il s'agit d'analyser les données puis de construire un modèle de prédiction de Y.

## En bref

Les dix variables sont gaussiennes et corrélées entre elles, mais elles n'ont que **9 degrés de liberté**. On les ramène à 9 composantes indépendantes (blanchiment), puis on **tourne cette base** pour qu'un seul axe porte toute l'information sur Y. Y s'écrit alors comme une fonction 1D d'une seule combinaison linéaire des variables :

$$t = \beta^\top (X-\mu), \qquad Y = f(t)$$

Ici, $f$ est une courbe lisse avec un coude, modélisée par une spline cubique.

| Approche | Modèle | Paramètres | MSE test |
|---|---|---:|---:|
| Première approche | MLP | 625 | 2.1e-5 |
| **Rotation de base** | **blanchiment + rotation supervisée + spline 1D** | **58** | **1.4e-6** |

L'erreur obtenue correspond au **plancher de bruit** des données : $R^2_{test} = 0.99999994$, et la validation croisée à 5 plis donne une MSE de $1.41\times10^{-6} \pm 0.02\times10^{-6}$.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| [`regression_rotation.ipynb`](regression_rotation.ipynb) | **Notebook principal** : analyse, rotation de base, modèle, validation, résidus |
| [`basis_rotation.py`](basis_rotation.py) | Estimateur scikit-learn `RotatedSingleIndexRegressor` et fonctions utilitaires |
| `regression.csv` | Données |
| `analysis.ipynb`, `regression_mlp.ipynb`, `model*.pth`, `run/` | Première approche : analyse exploratoire, XGBoost et MLP (PyTorch) |
| `requirements.txt` | Dépendances du notebook principal |

## Démarche

1. **Analyse rapide.** Les $X_i$ sont gaussiennes (asymétrie et kurtosis ≈ 0). Y est bornée dans $[0.551, 18.406]$ et son histogramme a trois pics. Ce n'est **pas un écrêtage** : on ne trouve que 5 valeurs exactement au minimum et 2 au maximum.
2. **Inter-corrélations, via le spectre et pas seulement les corrélations deux à deux.** La 10e valeur propre de la covariance vaut ≈ 4e-7, contre ≈ 8 pour la première. Une **contrainte linéaire exacte** relie les variables : X9 est une combinaison des 9 autres, avec $R^2 = 0.999997$. Les coefficients d'un modèle linéaire en X ne sont donc pas identifiables.
3. **Blanchiment.** $Z = \Lambda^{-1/2}V^\top(X-\mu)$ donne 9 composantes décorrélées, donc indépendantes puisque les données sont gaussiennes. Mais pour des données gaussiennes, **toute rotation** de $Z$ reste indépendante. La PCA et l'ICA ne voient que X et ne peuvent pas désigner la « bonne » base : c'est Y qui doit choisir la rotation.
4. **Rotation supervisée.** La *Sliced Inverse Regression* (SIR) donne une seule valeur propre non nulle (0.525, puis toutes < 0.001) : **Y ne dépend que d'une direction**. On complète cette direction en une base orthonormée (QR), puis on l'affine par Gauss-Newton. Dans la base tournée, l'axe 1 porte une courbe parfaite et les 8 autres axes sont du bruit indépendant de Y.
5. **Forme de $f$.**
   - Les trois pics de Y sont les extrema de $f$ ($f'=0$ en $t\approx-2.53$, $-0.11$ et $2.34$). Par changement de variable, $p_Y(y)=\sum_k \varphi(t_k)/|f'(t_k)|$ diverge en ces points.
   - $f$ présente un **coude** en $t\approx0.275$ ($y\approx3.0$) : elle reste continue, mais sa pente passe de 14.6 à 9.8.
   - Aucune forme analytique simple n'a été trouvée. $f$ est donc modélisée par une spline cubique (40 nœuds, nœud triple au coude, prolongement linéaire hors du domaine).
6. **Validation et résidus.** On utilise un découpage 70/15/15, une validation croisée à 5 plis et une comparaison à plusieurs modèles de référence. Les résidus sont proportionnels à $|f'(t)|$ : c'est un bruit gaussien sur l'indice, $y=f(t+\varepsilon)$ avec $\sigma_t\approx1.5\times10^{-4}$. La MSE attendue d'un prédicteur parfait, $\sigma_t^2\,\mathbb{E}[f'(t)^2]$, correspond à la MSE obtenue.

## Résultats (même découpage, jeu de test de 15 000 points)

| Modèle | MSE test |
|---|---:|
| Régression linéaire (X) | 1.8e+1 |
| Polynôme de degré 2 (X) | 6.0e+0 |
| Gradient boosting (X brut) | 2.5e-1 |
| Gradient boosting (t seul) | 8.8e-3 |
| Rotation + spline, sans coude (40 nœuds) | 2.6e-4 |
| **Rotation + spline + coude (40 nœuds)** | **1.4e-6** |

Rappel de la première approche (README précédent, **autre découpage**) : XGBoost obtenait une MSE test de 1.6e-1, le MLP à 9 729 paramètres 3.1e-5 et le MLP à 625 paramètres 2.1e-5.

## Utilisation

```bash
pip install -r requirements.txt
jupyter notebook regression_rotation.ipynb
```

```python
from basis_rotation import RotatedSingleIndexRegressor

model = RotatedSingleIndexRegressor(n_knots=40, kinks="auto").fit(X_train, y_train)
y_pred = model.predict(X_test)
t = model.index(X_test)          # coordonnée sur l'axe informatif
Z_rot = model.transform(X_test)  # 9 composantes indépendantes, colonne 0 = axe informatif
model.coef_, model.kinks_, model.sir_eigvals_
```

## Ce que cela change par rapport à la première analyse

| Hypothèse de la première analyse | Constat avec la rotation de base |
|---|---|
| Variables « peu corrélées » | Une contrainte linéaire exacte : X n'a que 9 degrés de liberté |
| Bornes de Y dues à un *clipping* artificiel | Les bornes sont les extrema de $f$ |
| Réduction de dimension non efficace (PCA) | La PCA, non supervisée, ne voit pas Y. Une réduction **supervisée** (SIR) ramène le problème à **1 dimension** |
| Forte non-linéarité en dimension 10 | La non-linéarité est entièrement contenue dans une fonction 1D |
| « 3 points particuliers autour de 0, 10 et 18 » | Ce sont les points stationnaires de $f$ |
| « Quelle ingénierie de variables pour un modèle plus petit ? » | $t=\beta^\top(X-\mu)$ : 58 paramètres au lieu de 625 |

## Questions ouvertes

- **Forme analytique de $f$** et origine du coude en $t\approx0.275$ : une définition par morceaux dans le générateur ?
- **Origine du bruit** : $\sigma_t\approx1.5\times10^{-4}$ sur l'indice et ≈ 6e-4 le long de la direction contrainte de X.
- **Extrapolation** : $t$ est observé dans $[-4.9, 4.2]$. Au-delà, le prolongement linéaire de $f$ est une hypothèse.
- **Robustesse** : SIR suppose une loi elliptique de X. Si la loi de X changeait, il faudrait réestimer la direction.
