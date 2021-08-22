import numpy as np

from sklearn.datasets import  fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile

from auto_cpca import AutoCPCA

X, y = fetch_olivetti_faces(return_X_y=True)

cpca_parameters = {'AutoCPCA__n_components': [5, 50, None],
                   'AutoCPCA__alpha': [.1, 1, 5],
                   'SVC__C': [1, 10]}
pca_parameters = {'PCA__n_components': [5, 50, None],
                  'SVC__C': [1, 10]}
fs_parameters = {'FS__percentile': [10, 30, 80],
                 'SVC__C': [1, 10]}
svc_parameters = {'C': [1, 10]}

svc = SVC()
auto_cpca_pipe = Pipeline([("AutoCPCA", AutoCPCA(verbose=0)), ("SVC", SVC())])
pca_pipe = Pipeline([("PCA", PCA()), ("SVC", SVC())])
fs_pipe = Pipeline([("FS", SelectPercentile()), ("SVC", SVC())])

cv = StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42)

# -- AutoPCA -> SVC --
clf = GridSearchCV(auto_cpca_pipe, cpca_parameters, cv=cv)
clf.fit(X, y)
cpca_max = np.max(clf.cv_results_["mean_test_score"])
cpca_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- PCA -> SVC --
clf = GridSearchCV(pca_pipe, pca_parameters, cv=cv)
clf.fit(X, y)
pca_max = np.max(clf.cv_results_["mean_test_score"])
pca_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- FeatureSelection -> SVC --
clf = GridSearchCV(fs_pipe, fs_parameters, cv=cv)
clf.fit(X, y)
fs_max = np.max(clf.cv_results_["mean_test_score"])
fs_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- plain SVC --
clf = GridSearchCV(svc, svc_parameters, cv=cv)
clf.fit(X, y)
svc_max = np.max(clf.cv_results_["mean_test_score"])
svc_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

print(f"AutoCPCA-SVC-pipeline maximal performance: {round(cpca_max, 4)} with std: {round(cpca_std, 4)}")
print(f"PCA-SVC-pipeline maximal performance: {round(pca_max, 4)} with std: {round(pca_std, 4)}")
print(f"FS-SVC-pipeline maximal performance: {round(fs_max, 4)} with std: {round(fs_std, 4)}")
print(f"SVC-pipeline maximal performance: {round(svc_max, 4)} with std: {round(svc_std, 4)}")
