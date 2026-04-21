from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.linalg import eigh


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=8):
        if n_components % 2 == 1:
            raise ValueError("n_components has to be even number")
        self.n_components = n_components

    def fit(self, X, y):
        # X (n_epochs, n_channels, n_times)
        # y (n_epochs)
        # 1. seperate two class
        cat0 = X[y == 0]
        cat1 = X[y == 1]
        # 2. calculate mean cov matrix

        cov0_list = []
        for trial in cat0:
            cov0_list.append(trial @ trial.T / trial.shape[1])
        cov0 = np.stack(cov0_list, axis=0).mean(axis=0)

        cov1_list = []
        for trial in cat1:
            cov1_list.append(trial @ trial.T / trial.shape[1])
        cov1 = np.stack(cov1_list, axis=0).mean(axis=0)

        eigvals, eigvecs = eigh(cov0, cov1)
        n_half = self.n_components // 2

        # index for max λ
        max_indices = np.argsort(eigvals)[-n_half:]
        # index for min λ
        min_indices = np.argsort(eigvals)[:n_half]

        # feature matrix
        self.W_ = np.hstack((eigvecs[:, max_indices], eigvecs[:, min_indices]))
        return self

    def transform(self, X):

        # X: (n_trials, n_channels, n_times)
        # 1. use W to filter each trial
        # 2. calculate variance and log
        # return: (n_trials, n_components)

        n_trials = X.shape[0]
        features = np.zeros((n_trials, self.n_components))

        for i, trial in enumerate(X):
            # trial: (n_channels, n_times)
            X_filtered = self.W_.T @ trial       # (n_components, n_times)
            features[i, :] = np.log(np.var(X_filtered, axis=1)) 

        return features
