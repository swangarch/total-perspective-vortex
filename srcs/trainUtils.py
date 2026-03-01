from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from mne.decoding import CSP
import os


# def train(X, y):
#     pipe = Pipeline([
#         ("csp", CSP(n_components=10, log=True)),
#         ("scaler", StandardScaler()),
#         ("clf", LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000)),
#     ])
#     print("Training with cross validation.")
#     scores = cross_val_score(pipe, X, y, cv=2)
#     print("Train score", scores.mean())

#     print("Training save weights.")
#     pipe.fit(X, y)
#     return pipe, scores.mean()


def train(X, y):
    pipe = Pipeline([
        ("csp", CSP(n_components=8, log=True)),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16, 4),
            activation="relu",
            max_iter=5000,
            solver="adam",
            random_state=422,
            batch_size=32,
            alpha=0.001,
            tol=0,
            learning_rate="adaptive",
            learning_rate_init=0.005,
            early_stopping=True,
            shuffle=True,
            n_iter_no_change=5000
        ))
    ])
    print("Training with cross validation.")
    scores = cross_val_score(pipe, X, y, cv=2)
    print("Train score", scores.mean())

    print("Training save weights.")
    pipe.fit(X, y)
    plt.plot(pipe["mlp"].loss_curve_)
    plt.show()
    return pipe, scores.mean()
