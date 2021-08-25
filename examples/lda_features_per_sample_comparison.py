# adapted example from sklearn:
# https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html#sphx-glr-auto-examples-classification-plot-lda-py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from auto_cpca import AutoCPCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y


acc_clf1, acc_clf2, acc_clf3, acc_clf4 = [], [], [], []
n_features_range = range(2, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2, score_clf3, score_clf4 = 0, 0, 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        clf1 = LinearDiscriminantAnalysis(solver='lsqr',
                                          shrinkage='auto').fit(X, y)
        clf2 =Pipeline([("AutoCPCA", AutoCPCA(n_components=min(n_features, 5), alpha=1)), ("SVC", SVC())]).fit(X, y)
        clf3 = QuadraticDiscriminantAnalysis().fit(X, y)
        clf4 = SVC().fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)
        score_clf3 += clf3.score(X, y)
        score_clf4 += clf4.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)
    acc_clf3.append(score_clf3 / n_averages)
    acc_clf4.append(score_clf4 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
         label="LDA with Ledoit Wolf", color='navy')
plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
         label="AutoCPCA-SVC", color='gold')
plt.plot(features_samples_ratio, acc_clf3, linewidth=2,
         label="SVC", color='red')
plt.plot(features_samples_ratio, acc_clf4, linewidth=2,
         label="QDA", color='green')

plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')

plt.legend(loc=3, prop={'size': 12})
plt.suptitle('Shrinkage LDA vs. '
             + 'AutCPCA+SVC vs. '
             + 'QDA')
plt.tight_layout()
plt.show()