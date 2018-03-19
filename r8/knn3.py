"""%%file word_movers_knn.py"""

# Authors: Vlad Niculae, Matt Kusner
# License: Simplified BSD

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize

class WordMoversKNN(KNeighborsClassifier):
    _pairwise = False

    def __init__(self, word_dis,n_neighbors=1, n_jobs=1, verbose=False):
        self.word_dis = word_dis
        super(WordMoversKNN, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs,
                                            metric='precomputed', algorithm='brute')

    def _wmd_row(self, row, X_train):
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        n_samples_test = X_test.shape[0]

        if X_train is None:
            X_train = self._fit_X

        #dist=[self._wmd_row(i,X_train) for i in X_test.shape[0]] #i th
        dist1 = self.word_dis
        #print '*****:',dist1.shape
        return np.array(dist1)

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WordMoversKNN, self).fit(X, y)

    def predict(self, X):
       
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WordMoversKNN, self).predict(dist)
