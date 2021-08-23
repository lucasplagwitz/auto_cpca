import unittest
import warnings
import numpy as np
from auto_cpca import AutoCPCA

from sklearn.datasets import load_iris, load_boston, fetch_olivetti_faces
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC


class TestAutoCPCA(unittest.TestCase):

    def test_sklearn_pipe(self):
        cpca_parameters = {'AutoCPCA__alpha': [.1, 1, 5],
                           'SVC__C': [1, 10]}
        auto_cpca_pipe = Pipeline([("AutoCPCA", AutoCPCA(verbose=0)), ("SVC", SVC())])

        # -- AutoPCA -> SVC --
        clf = GridSearchCV(auto_cpca_pipe, cpca_parameters,
                           cv=StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42))

        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        cpca_max = np.max(clf.cv_results_["mean_test_score"])
        cpca_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]
        self.assertGreater(cpca_max, 0.95)
        self.assertLess(cpca_std, 0.05)

    def test_alpha(self):
        X, y = load_iris(return_X_y=True)
        _ = AutoCPCA(alpha=5).fit(X, y)

        _ = AutoCPCA(alpha={0: 3, 1: 5, 2: 0}).fit(X, y)

        with self.assertRaises(ValueError):
            _ = AutoCPCA(alpha=(1, 2, 3)).fit(X, y)

        with self.assertRaises(ValueError):
            _ = AutoCPCA(alpha={100: 3, 1:5}).fit(X, y)

        with self.assertRaises(NotImplementedError):
            _ = AutoCPCA(alpha="auto").fit(X, y)

    def test_fit(self):
        X, y = load_iris(return_X_y=True)
        auto_cpca = AutoCPCA().fit(X, y)
        self.assertTrue(isinstance(auto_cpca, AutoCPCA))
        check_is_fitted(auto_cpca)

    def test_fit_transform(self):
        X, y = load_iris(return_X_y=True)
        auto_cpca = AutoCPCA(n_components=2).fit(X, y)

        X_trans = auto_cpca.transform(X)
        assert np.shape(X_trans)[1] == 2

        X_trans_direct = AutoCPCA(n_components=2).fit_transform(X, y)

        assert np.array_equal(X_trans, X_trans_direct)

    def test_classification(self):
        X, y = load_boston(return_X_y=True)
        with self.assertRaises(ValueError):
            AutoCPCA().fit(X, y)
        with self.assertRaises(ValueError):
            AutoCPCA().fit_transform(X, y)

        X, y = load_iris(return_X_y=True)
        classes = np.unique(y)
        auto_cpca = AutoCPCA().fit(X, y)
        assert np.array_equal(classes, auto_cpca.classes_)

    def test_feature_influence(self):
        X, y = load_iris(return_X_y=True)
        auto_cpca = AutoCPCA(n_components=2).fit(X, y)

        back_mapped = auto_cpca.feature_influence()
        assert np.shape(back_mapped) == (4, )
        back_mapped2 = auto_cpca.feature_influence(np.array([0.2, 0.8]))
        assert np.shape(back_mapped2) == (4,)
        assert ~ np.array_equal(back_mapped, back_mapped2)

    def test_pca_preprocessing(self):
        X, y = fetch_olivetti_faces(return_X_y=True)

        auto_cpca = AutoCPCA(n_components=5).fit(X, y)

        assert auto_cpca.pca is not None
        assert auto_cpca.transform(X).shape == (X.shape[0], 5)
        assert auto_cpca.inverse_transform(auto_cpca.transform(X)).shape == X.shape
        assert auto_cpca.feature_influence().shape == (X.shape[1], )

    def test_norm_cov_mean(self):
        X, y = load_iris(return_X_y=True)
        with warnings.catch_warnings(record=True) as w:
            _ = AutoCPCA(norm_cov_mean=True).fit(X, y)
            assert "The benefit of norm_cov_mean" in str(w[-1].message)