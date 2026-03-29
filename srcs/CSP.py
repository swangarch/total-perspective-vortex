from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class my_CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_comp=8):
        self.n_comp = n_comp

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self