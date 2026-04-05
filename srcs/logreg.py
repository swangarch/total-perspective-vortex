from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np


def sigmoid(X):
    s = 1.0 / (1 + np.e ** (-X))
    return s


class MyLogreg(BaseEstimator, ClassifierMixin):
    def __init__(self, n_comp=8, learning_rate = 0.001,
                 max_iter = 1000, batch_size = 24,
                 C = 0.1):
        self.n_comp = n_comp
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.C = C
        self.lam = 1.0 / C

    def fit(self, X, y): # X (n_samples, n_features)
        self.W_ = np.zeros((X.shape[1], 1))
        self.b_ = 0
        y = y.reshape((-1, 1))
        for i in range(self.max_iter):
            p = sigmoid(X @ self.W_ + self.b_)
            d = (p - y) / float(len(X))
            grad_b = d.T.sum()
            grad_W = X.T @ d + (self.lam * self.W_) / len(X)
            self.W_ -= self.learning_rate * grad_W
            self.b_ -= self.learning_rate * grad_b
        return self
    
    def predict(self, X):
        prediction = (sigmoid(X @ self.W_ + self.b_) > 0.5).astype(int).flatten()
        return prediction
