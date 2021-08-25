import numpy as np

from sklearn.datasets import  fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import chi2, f_classif

from auto_cpca import AutoCPCA

X, y = fetch_olivetti_faces(return_X_y=True)

cpca_parameters = {'AutoCPCA__n_components': [5, 50, 100, None],
                   'AutoCPCA__alpha': [.1, 1, 5]}
pca_parameters = {'PCA__n_components': [5, 50, 100, 200, None],
                  "SVC__C": [.1, 1, 10]}
fs_parameters = {'FS__percentile': [5, 10, 30, 60, 80],
                 'FS__score_func': [chi2, f_classif],
                 "SVC__C": [.1, 1, 10]}
svc_parameters = {"C": [.1, 1, 10]}

svc = SVC(decision_function_shape="ovr")
auto_cpca_pipe = Pipeline([("AutoCPCA", AutoCPCA(verbose=0)), ("SVC", SVC())])
pca_pipe = Pipeline([("PCA", PCA()), ("SVC", SVC())])
lda_pipe_01 = Pipeline([("LDA", LinearDiscriminantAnalysis()), ("SVC", SVC())])
lda_pipe_02 = Pipeline([("PCA", PCA()), ("LDA", LinearDiscriminantAnalysis()), ("SVC", SVC())])
lda_pipe_03 = Pipeline([("LDA", LinearDiscriminantAnalysis())])
qda_pipe = Pipeline([("QDA", QuadraticDiscriminantAnalysis())])
fs_pipe = Pipeline([("FS", SelectPercentile()), ("SVC", SVC())])

cv = StratifiedShuffleSplit(n_splits=15, test_size=.2, random_state=42)

# -- AutoCPCA -> SVC --
clf = GridSearchCV(auto_cpca_pipe, cpca_parameters, cv=cv, n_jobs=8)
clf.fit(X, y)
cpca_max = np.max(clf.cv_results_["mean_test_score"])
cpca_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- PCA -> SVC --
clf = GridSearchCV(pca_pipe, pca_parameters, cv=cv, n_jobs=8)
clf.fit(X, y)
pca_max = np.max(clf.cv_results_["mean_test_score"])
pca_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- FeatureSelection -> SVC --
clf = GridSearchCV(fs_pipe, fs_parameters, cv=cv, n_jobs=8)
clf.fit(X, y)
fs_max = np.max(clf.cv_results_["mean_test_score"])
fs_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- LDA --
lda_max_all = lda_max = 0
best_lda = ""
for num in ["01", "02", "03"]:
    clf = GridSearchCV(eval("lda_pipe_"+num), {}, cv=cv, n_jobs=8)
    clf.fit(X, y)
    lda_max = np.max(clf.cv_results_["mean_test_score"])
    lda_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]
    if lda_max_all < lda_max:
        lda_max_all = lda_max
        best_lda = num
print(f"Best LDA-Pipe: {best_lda}")


# -- QDA --
clf = GridSearchCV(qda_pipe, {}, cv=cv, n_jobs=8)
clf.fit(X, y)
qda_max = np.max(clf.cv_results_["mean_test_score"])
qda_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

# -- plain SVC --
clf = GridSearchCV(svc, svc_parameters, cv=cv, n_jobs=8)
clf.fit(X, y)
svc_max = np.max(clf.cv_results_["mean_test_score"])
svc_std = clf.cv_results_["std_test_score"][np.argmax(clf.cv_results_["mean_test_score"])]

print(f"AutoCPCA-SVC-pipeline maximal performance: {round(cpca_max, 4)} with std: {round(cpca_std, 4)}")
print(f"PCA-SVC-pipeline maximal performance: {round(pca_max, 4)} with std: {round(pca_std, 4)}")
print(f"FS-SVC-pipeline maximal performance: {round(fs_max, 4)} with std: {round(fs_std, 4)}")
print(f"LDA-pipeline maximal performance: {round(lda_max, 4)} with std: {round(lda_std, 4)}")
print(f"QDA-pipeline maximal performance: {round(qda_max, 4)} with std: {round(qda_std, 4)}")
print(f"SVC-pipeline maximal performance: {round(svc_max, 4)} with std: {round(svc_std, 4)}")
