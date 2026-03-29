from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class my_logreg(BaseEstimator, TransformerMixin):
    def __init__(self, n_comp=8):
        self.n_comp = n_comp

    def fit(self, X, y): # X (n_samples, n_features)
        return self

    def transform(self, X):
        return self
    
    def predict(self, X):
        pass
        # return None