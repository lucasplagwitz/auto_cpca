import warnings
from typing import Union
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score


class AutoCPCA(TransformerMixin, BaseEstimator):
    """
    Automatic Contrastive Principle Components (AutoCPCA).

    preprocess_with_pca_dim: int
            If this parameter is provided (and it is greater than n_features), then both the foreground and background
            datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
            but n_features > 1000, a preliminary round of PCA is automatically performed to reduce the
            dimensionality to 1000.
    """

    def __init__(self, n_components: int = None, alpha: Union[int, float, dict, str] = 1,
                 preprocess_with_pca_dim: int = 1000, norm_cov_mean: bool = False, verbose: int = 1):

        self.n_components = n_components
        self.preprocess_with_pca_dim = preprocess_with_pca_dim
        self.verbose = verbose
        self.alpha = alpha
        self.classes_ = None

        self.fg_cov = None
        self.bg_cov = {}
        self.v_top = None
        self.pca = None
        self.norm_cov_mean = norm_cov_mean

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray):

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        if isinstance(self.alpha, dict) and len(self.alpha) != len(self.classes_):
            raise ValueError("Length missmatched")

        if self.n_components is None:
            self.n_components = min(np.min(X.shape), self.preprocess_with_pca_dim) - 1

        fg, bgs = self.divide(X, y)  # backgrounds: list of n_classes

        if fg.shape[1] > self.preprocess_with_pca_dim:
            max_dimensions = np.min([len(fg), self.preprocess_with_pca_dim])
            self.pca = PCA(n_components=max_dimensions)
            fg = self.pca.fit_transform(fg)
            for key in bgs.keys():
                bgs[key] = self.pca.transform(bgs[key])

            if self.verbose:
                print("Data dimensionality reduced to " + str(max_dimensions) + ". Percent variation retained: ~" +
                      str(int(100 * np.sum(self.pca.explained_variance_ratio_))) + '%')

        # Calculate the covariance matrices
        for key in bgs.keys():
            self.bg_cov[key] = np.cov(bgs[key].T)
        self.fg_cov = np.cov(fg.T)

        if self.norm_cov_mean:
            warnings.warn("The benefit of norm_cov_mean has not been considered.")
            for key in self.bg_cov.keys():
                self.bg_cov[key] = self.bg_cov[key] - self.bg_cov[key].mean(axis=0)
            self.fg_cov = (self.fg_cov - self.fg_cov.mean(axis=0))

        if self.alpha == "auto":
            msg = "Not yet."
            alpha = self.optimal_alpha(X, y)
        elif isinstance(self.alpha, dict):
            alpha = self.alpha
        elif isinstance(self.alpha, (int, float)):
            alpha = {label: self.alpha for label in self.classes_}
        self.create_component(alpha=alpha)

        return self

    def transform(self, X, y=None):

        check_is_fitted(self)

        # transform - feature selection via PCA for many features
        if self.pca is not None and X.shape[1] > self.preprocess_with_pca_dim:
            X = self.pca.transform(X)
        reduced_dataset = X.dot(self.v_top)
        return reduced_dataset

    def inverse_transform(self, X, y=None):
        X = X.dot(self.v_top.T)
        if self.pca is not None:
            X = self.pca.inverse_transform(X)
        return X

    def create_component(self, alpha: np.ndarray):

        sigma = self.fg_cov - sum([alpha[key]*val for key, val in self.bg_cov.items()])
        w, v = np.linalg.eigh(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        self.v_top = np.real(v[:, eig_idx])  # sigma is quasi symm. -> imaginary part only numerical reasons

    def divide(self, X, y):
        bgs = {}
        for label in self.classes_:
            bgs[label] = X[y == label]
        return X, bgs

    def optimal_alpha(self, X, y, metric=balanced_accuracy_score, start_value=1, n_iter=500):
        msg = "Not ready yet - please use hyperparameter optimization for the alpha-choice. " \
              "A suitable selection-algorithm will follow in the future. "
        raise NotImplemented(msg)

        if np.issubdtype(self.alpha, np.integer):
            alpha = np.array([self.alpha] * len(self.classes_))
        elif isinstance(self.alpha, np.ndarray):
            alpha = self.alpha
        first_val = None
        for iter in range(n_iter):
            self.create_component(alpha)
            dtc = DecisionTreeClassifier()
            dtc.fit(X, y)
            val = metric(y_true=y, y_pred=dtc.predict(X))

            if val < 0.99:
                if iter == 0:
                    first_val = "lower"
                else:
                    if first_val == "one":
                        return alpha
                alpha += 0.1*np.min(alpha)
            else:
                if iter == 0:
                    first_val = "one"
                else:
                    if first_val == "lower":
                        return alpha
                alpha -= 0.1 * np.min(alpha)

        msg = "Not converged"
        warnings.warn(msg)
        return alpha

    def feature_influence(self, weight=1):
        check_is_fitted(self)
        return np.sum(weight*np.abs(self.v_top), axis=1)
