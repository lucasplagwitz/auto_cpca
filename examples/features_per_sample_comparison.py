# adapted example from sklearn:
# https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html#sphx-glr-auto-examples-classification-plot-lda-py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectPercentile

from auto_cpca import AutoCPCA

train_test_ratio = 0.5
n_samples = [200, 500]  # samples for testing
n_averages = 20  # how often to repeat classification
n_features_max = 2000  # maximum number of features
step = 150  # step size for the calculation

fig, ax = plt.subplots(1, 2)
for num_c, n_samples in enumerate([200, 500]):
    acc_clf = [[],[],[],[],[]]
    scoring_clf = [[],[],[],[],[]]
    n_features_range = range(100, n_features_max + 1, step)
    for n_features in n_features_range:
        print(f"-- {n_features} --")
        scoring_clf = [[],[],[],[],[]]
        for i in range(n_averages):
            clf0 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf1 = Pipeline(
                [("AutoCPCA", AutoCPCA(n_components=min(n_features // 20, 20), preprocess_with_pca_dim=1000)),
                 ("SVC", SVC())])
            clf2 = Pipeline([("FS", SelectPercentile()), ("SVC", SVC())])
            clf3 = SVC()
            clf4 = Pipeline([("PCA", PCA()), ("SVC", SVC())])
            X, y = make_classification(n_samples, n_features,
                                                 n_informative=n_features//5,
                                                 n_redundant=0,
                                                 n_repeated=0,
                                                 n_clusters_per_class=3,
                                                 hypercube=False,
                                                 n_classes=8,
                                                 class_sep=40,
                                                 random_state=i,
                                                 )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio)

            for num in range(5):
                clf = eval(f"clf{num}")
                clf.fit(X_train, y_train)
                scoring_clf[num].append(clf.score(X_test, y_test))
        for num in range(5):
            acc_clf[num].append(np.mean(scoring_clf[num]))

    features_samples_ratio = np.array(n_features_range) / int(n_samples*train_test_ratio)

    ax[num_c].plot(features_samples_ratio, acc_clf[0], linewidth=2, color='navy')
    ax[num_c].plot(features_samples_ratio, acc_clf[1], linewidth=2, color='gold')
    ax[num_c].plot(features_samples_ratio, acc_clf[2], linewidth=2, color='red')
    ax[num_c].plot(features_samples_ratio, acc_clf[3], linewidth=2, color='green')
    ax[num_c].plot(features_samples_ratio, acc_clf[4], linewidth=2, color='purple')

    if num_c == 0:
        ax[num_c].set_ylabel('Classification accuracy')

    ax[num_c].set_xlabel('n_features / n_samples')

    ax[num_c].set_title(f"{int(n_samples*train_test_ratio)} train samples")

plt.subplots_adjust(bottom=0.3, wspace=0.33)

plt.legend(labels=["LDA with Ledoit Wolf", 'AutoCPCA->SVC', 'FS->SVC', 'SVC', 'PCA->SVC'],loc='upper center',
             bbox_to_anchor=(-.15, -0.2), fancybox=True, shadow=True, ncol=3)
plt.savefig("../demo/performance_feature_sample_ratio.png")
