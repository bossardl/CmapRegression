"""Rotation de base + modèle à indice unique (single-index model).

Idée : les variables X sont gaussiennes et inter-corrélées. On les ramène à des
composantes indépendantes (centrage + PCA + blanchiment), puis on choisit, parmi
toutes les rotations possibles de cette base blanchie, celle dont le premier axe
porte toute l'information sur Y (Sliced Inverse Regression, affinée par
Gauss-Newton). Y est alors modélisé par une fonction 1D (spline cubique) de ce
seul axe :

    t = beta^T (x - mu)          (projection apprise)
    y = f(t)                     (spline cubique 1D, éventuellement avec un "coude")
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


# --------------------------------------------------------------------------- #
# Briques élémentaires
# --------------------------------------------------------------------------- #
def whitening_matrix(X: np.ndarray, rank_tol: float = 1e-6):
    """Moyenne, spectre et matrice de blanchiment de X.

    Les directions de variance quasi nulle (valeur propre < rank_tol * max) sont
    écartées : elles traduisent une contrainte linéaire exacte entre variables
    et ne portent aucune information.
    """
    mean = X.mean(axis=0)
    cov = np.cov(X - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = eigvals > rank_tol * eigvals[0]
    W = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    return mean, eigvals, eigvecs, W


def sliced_inverse_regression(Z: np.ndarray, y: np.ndarray, n_slices: int = 50):
    """SIR (Li, 1991) sur des données déjà blanchies.

    On découpe y en tranches de même effectif et on fait une PCA des moyennes
    conditionnelles E[Z | y dans la tranche]. Si y = f(u^T Z), ces moyennes
    sont toutes alignées sur u : une seule valeur propre non nulle.
    """
    edges = np.quantile(y, np.linspace(0, 1, n_slices + 1))
    slice_id = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, n_slices - 1)
    M = np.zeros((Z.shape[1], Z.shape[1]))
    for k in range(n_slices):
        mask = slice_id == k
        if mask.any():
            m = Z[mask].mean(axis=0)
            M += mask.mean() * np.outer(m, m)
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def complete_rotation(u: np.ndarray) -> np.ndarray:
    """Matrice orthogonale dont la première colonne est u (complétion par QR)."""
    k = len(u)
    Q, _ = np.linalg.qr(np.column_stack([u, np.eye(k)]))
    Q = Q[:, :k]
    if Q[:, 0] @ u < 0:
        Q[:, 0] *= -1
    return Q


def slope_jumps(t: np.ndarray, y: np.ndarray, lo: float = -3.0, hi: float = 3.0,
                step: float = 0.0025, half_width: float = 0.03, min_points: int = 60):
    """Balayage des ruptures de pente de y = f(t).

    En chaque point c, on ajuste un polynôme de degré 2 à gauche et à droite
    de c et on compare les pentes en c. Renvoie (centres, saut de pente, z-score
    robuste). Un "coude" (f continue mais f' discontinue) donne un pic de |z|.
    """
    order = np.argsort(t)
    ts, ys = t[order], y[order]
    centers, jumps = [], []
    for c in np.arange(lo, hi, step):
        i0, i1, i2 = np.searchsorted(ts, [c - half_width, c, c + half_width])
        if i1 - i0 < min_points or i2 - i1 < min_points:
            continue
        pl = np.polyfit(ts[i0:i1] - c, ys[i0:i1], 2)
        pr = np.polyfit(ts[i1:i2] - c, ys[i1:i2], 2)
        centers.append(c)
        jumps.append(pr[1] - pl[1])
    centers, jumps = np.array(centers), np.array(jumps)
    med = np.median(jumps)
    mad = 1.4826 * np.median(np.abs(jumps - med))
    return centers, jumps, (jumps - med) / mad


def detect_kinks(t: np.ndarray, y: np.ndarray, z_threshold: float = 50.0,
                 half_width: float = 0.03, **kw) -> list[float]:
    """Positions approximatives des coudes (pics de |z| isolés au-delà du seuil)."""
    centers, _, z = slope_jumps(t, y, half_width=half_width, **kw)
    kinks: list[float] = []
    for i in np.argsort(-np.abs(z)):
        if abs(z[i]) < z_threshold:
            break
        if all(abs(centers[i] - k) > 2 * half_width for k in kinks):
            kinks.append(float(centers[i]))
    return sorted(kinks)


class LinearlyExtrapolatedSpline:
    """Spline cubique de régression, prolongée linéairement hors de [a, b].

    Les nœuds intérieurs sont uniformément espacés entre les quantiles
    `tail` et 1 - `tail` de t : on garde de la résolution dans les queues
    (contrairement à des nœuds placés aux quantiles). Chaque coude reçoit un
    nœud triple, ce qui autorise une discontinuité de f' à cet endroit.
    """

    def __init__(self, t: np.ndarray, y: np.ndarray, n_knots: int,
                 kinks=(), tail: float = 5e-4):
        order = np.argsort(t)
        ts, ys = t[order], y[order]
        self.a, self.b = float(ts[0]), float(ts[-1])
        lo, hi = np.quantile(ts, [tail, 1 - tail])
        interior = np.linspace(lo, hi, n_knots)
        spacing = (hi - lo) / max(n_knots - 1, 1)
        for k in kinks:  # on remplace les nœuds voisins par un nœud triple
            interior = interior[np.abs(interior - k) > spacing / 2]
            interior = np.concatenate([interior, [k, k, k]])
        interior = np.sort(interior)
        knots = np.concatenate([[self.a] * 4, interior, [self.b] * 4])
        self.knots = knots
        self.kinks = tuple(kinks)
        self.bspline = make_lsq_spline(ts, ys, knots, k=3)
        self.dspline = self.bspline.derivative()

    @property
    def n_coefficients(self) -> int:
        return len(self.bspline.c)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, self.a, self.b)
        out = self.bspline(tc)
        out = np.where(t < self.a, self.bspline(self.a) + self.dspline(self.a) * (t - self.a), out)
        out = np.where(t > self.b, self.bspline(self.b) + self.dspline(self.b) * (t - self.b), out)
        return out

    def derivative(self, t):
        return self.dspline(np.clip(np.asarray(t, dtype=float), self.a, self.b))


# --------------------------------------------------------------------------- #
# Estimateur scikit-learn
# --------------------------------------------------------------------------- #
class RotatedSingleIndexRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """Blanchiment + rotation supervisée + spline 1D.

    Paramètres
    ----------
    n_knots : nombre de nœuds intérieurs de la spline f.
    kinks : None (spline lisse), "auto" (détection des coudes sur le jeu
        d'apprentissage) ou liste de positions approximatives sur l'axe t.
    n_slices : nombre de tranches de y pour la SIR.
    n_refine : itérations de Gauss-Newton pour affiner la direction.
    rank_tol : seuil relatif pour écarter les directions de variance nulle.

    Attributs après `fit`
    ---------------------
    mean_, eigvals_ : moyenne et spectre de la covariance de X.
    whitening_ : (p, k) passage de X centré à la base blanchie Z.
    sir_eigvals_ : spectre SIR (combien de directions comptent pour Y).
    direction_ : (k,) direction unitaire u dans la base blanchie.
    rotation_ : (k, k) rotation orthogonale dont la 1re colonne est u.
    coef_ : (p,) beta tel que t = (x - mean_) @ coef_ (beta de norme minimale).
    kinks_ : positions des coudes retenues (sur l'axe t).
    spline_ : la fonction f.
    """

    def __init__(self, n_knots: int = 120, kinks="auto", n_slices: int = 50,
                 n_refine: int = 5, rank_tol: float = 1e-6):
        self.n_knots = n_knots
        self.kinks = kinks
        self.n_slices = n_slices
        self.n_refine = n_refine
        self.rank_tol = rank_tol

    # -- étapes internes ---------------------------------------------------- #
    def _refine_direction(self, Z, y, u, kinks):
        """Gauss-Newton : y ≈ f(Z u)  =>  r ≈ f'(t) · Z du  (moindres carrés)."""
        for _ in range(self.n_refine):
            t = Z @ u
            spline = LinearlyExtrapolatedSpline(t, y, self.n_knots, kinks)
            r = y - spline(t)
            self.refine_history_.append(float(np.mean(r ** 2)))
            J = spline.derivative(t)[:, None] * Z
            u = u + np.linalg.lstsq(J, r, rcond=None)[0]
            u /= np.linalg.norm(u)
        return u

    @staticmethod
    def _locate_kink(t, y, n_knots, guess, others, half_width=0.03):
        """Position du coude qui minimise l'erreur d'ajustement (recherche 1D)."""
        def loss(k):
            sp = LinearlyExtrapolatedSpline(t, y, n_knots, sorted([*others, k]))
            return np.mean((y - sp(t)) ** 2)
        res = minimize_scalar(loss, bounds=(guess - half_width, guess + half_width),
                              method="bounded", options={"xatol": 1e-5})
        return float(res.x)

    # -- API scikit-learn ----------------------------------------------------- #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_, self.eigvals_, self.eigvecs_, self.whitening_ = whitening_matrix(X, self.rank_tol)
        Z = (X - self.mean_) @ self.whitening_

        # 1) direction initiale : SIR
        self.sir_eigvals_, sir_dirs = sliced_inverse_regression(Z, y, self.n_slices)
        u = sir_dirs[:, 0]
        if np.corrcoef(Z @ u, y)[0, 1] < 0:  # convention : t croît avec y
            u = -u

        # 2) affinage sans coude, puis détection / localisation des coudes
        self.refine_history_ = []
        u = self._refine_direction(Z, y, u, kinks=())
        t = Z @ u
        if self.kinks is None:
            kinks = []
        elif isinstance(self.kinks, str) and self.kinks == "auto":
            kinks = detect_kinks(t, y)
        else:
            kinks = list(self.kinks)
        for i, k in enumerate(kinks):
            kinks[i] = self._locate_kink(t, y, self.n_knots, k, kinks[:i] + kinks[i + 1:])

        # 3) affinage final avec les coudes
        if kinks:
            u = self._refine_direction(Z, y, u, kinks)
            t = Z @ u

        self.kinks_ = sorted(kinks)
        self.direction_ = u
        self.rotation_ = complete_rotation(u)
        self.coef_ = self.whitening_ @ u
        self.spline_ = LinearlyExtrapolatedSpline(t, y, self.n_knots, self.kinks_)
        self.t_range_ = (float(t.min()), float(t.max()))
        return self

    def index(self, X) -> np.ndarray:
        """Coordonnée t = beta^T (x - mu) sur l'axe informatif."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.coef_

    def transform(self, X) -> np.ndarray:
        """Composantes indépendantes après rotation (colonne 0 = axe informatif)."""
        Z = (np.asarray(X, dtype=float) - self.mean_) @ self.whitening_
        return Z @ self.rotation_

    def predict(self, X) -> np.ndarray:
        return self.spline_(self.index(X))

    @property
    def n_parameters_(self) -> int:
        """Paramètres libres : beta (p) + moyenne projetée (1) + coefficients de spline + coudes."""
        return len(self.coef_) + 1 + self.spline_.n_coefficients + len(self.kinks_)
