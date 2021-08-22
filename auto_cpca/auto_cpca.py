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

    Parameters
    ----------
    n_components: int or None, default=None
        Number of components to keep.
        If n_components is None all components are kept::

            n_components == min(n_samples, n_features) - 1

    alpha: float, dict or str, default=1
        Background weighting parameter.

        If alpha is a number, every background will be penalized with this factor.

        If alpha is a dict, the backgrounds will be penalized individually.

        If alpha is set to 'auto', AutoCPCA calculate a suitable alpha by run the
        decomposition multiple times. Not implemented yet!

    preprocess_with_pca_dim: int, default=1000
        If this parameter is provided (and it is greater than n_features), then both the foreground and background
        datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
        but n_features > 1000, a preliminary round of PCA is automatically performed to reduce the
        dimensionality to 1000.

    norm_cov_mean: bool, default=False
        Normalize mean of covariance.

    verbose: int, default=1
        Print reduce message for PCA-based preprocessing.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data with respect to fore- and background.

    """
    def __init__(self, n_components: int = None, alpha: Union[int, float, dict, str] = 1,
                 preprocess_with_pca_dim: int = 1000, norm_cov_mean: bool = False, verbose: int = 1):
        self.n_components = n_components
        self.preprocess_with_pca_dim = preprocess_with_pca_dim
        self.verbose = verbose
        self.alpha = alpha
        self.fg_cov = None
        self.bg_cov = {}
        self.norm_cov_mean = norm_cov_mean
        self.pca = None

        self.classes_ = None
        self.components_ = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.

        """
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        if isinstance(self.alpha, dict) and len(self.alpha) != len(self.classes_):
            raise ValueError("Length missmatched")

        if self.n_components is None:
            self.n_components = min(np.min(X.shape), self.preprocess_with_pca_dim) - 1

        fg, bgs = self._divide(X, y)  # backgrounds: dict of n_classes

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
            alpha = self._optimal_alpha(X, y)
        elif isinstance(self.alpha, dict):
            alpha = self.alpha
        elif isinstance(self.alpha, (int, float)):
            alpha = {label: self.alpha for label in self.classes_}
        self._create_component(alpha=alpha)

        return self

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        y: ignored

        Returns
        -------
        reduced_dataset : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        # transform - feature selection via PCA for many features
        if self.pca is not None and X.shape[1] > self.preprocess_with_pca_dim:
            X = self.pca.transform(X)
        reduced_dataset = X.dot(self.components_)
        return reduced_dataset

    def feature_influence(self, weight=1):
        """Sums influence of features to principle components.

        Thought for feature importance.

        Parameters
        ----------
        weight: float or array with shape (n_components, )
            Multiply component by weight.

        Returns
        -------
            Weighted sum of positive feature-influence in principle component.
        """
        check_is_fitted(self)
        return np.sum(weight*np.abs(self.components_), axis=1)

    def inverse_transform(self, X, y=None):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        y: ignored

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self)
        X_original = X.dot(self.components_.T)
        if self.pca is not None:
            X_original = self.pca.inverse_transform(X_original)
        return X_original

    def _create_component(self, alpha: np.ndarray):
        """Apply SVD on sigma."""
        sigma = self.fg_cov - sum([alpha[key]*val for key, val in self.bg_cov.items()])
        w, v = np.linalg.eigh(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        self.components_ = np.real(v[:, eig_idx])  # sigma is quasi symm. -> imaginary part only numerical reasons

    def _divide(self, X, y):
        """Calculate foreground and backgrounds."""
        bgs = {}
        for label in self.classes_:
            bgs[label] = X[y == label]
        return X, bgs

    def _optimal_alpha(self, X, y, metric=balanced_accuracy_score, start_value=1, n_iter=500):
        """Optimal parameter-selection procedure."""
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
