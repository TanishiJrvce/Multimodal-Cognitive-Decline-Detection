"""Speech-state discovery: fit on train only; compare k; measure stability. No MCI naming."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score


def fit_states(Z_train, k, n_pca=10, seed=0):
    pca = PCA(n_components=min(n_pca, Z_train.shape[1]), random_state=seed).fit(Z_train)
    gmm = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed).fit(pca.transform(Z_train))
    return pca, gmm


def compare_k(Z_train, k_range=range(1, 7), seed=0, n_boot=50):
    """Per k: BIC, silhouette, bootstrap stability (ARI of labels on the full train set)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in k_range:
        pca, gmm = fit_states(Z_train, k, seed=seed)
        P = pca.transform(Z_train)
        base = gmm.predict(P)
        aris = []
        if k > 1:
            for _ in range(n_boot):
                idx = rng.choice(len(Z_train), len(Z_train))
                p2, g2 = fit_states(Z_train[idx], k, seed=int(rng.integers(1e6)))
                aris.append(adjusted_rand_score(base, g2.predict(p2.transform(Z_train))))
        out[k] = dict(bic=gmm.bic(P), silhouette=silhouette_score(P, base) if k > 1 else np.nan,
                      stability=float(np.mean(aris)) if aris else np.nan)
    return out


def order_states_by_progression(labels, prog_scores):
    """Report mean progression per state to tell *ordered stages* (monotone along progression and
    along within-person time) from *different profiles* (separated but not ordered)."""
    return {int(s): float(np.mean(prog_scores[labels == s])) for s in np.unique(labels)}
