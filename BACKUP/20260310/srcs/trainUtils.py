# from mne.io import concatenate_raws, read_raw_edf
# import matplotlib.pyplot as plt
# import mne
# import sys
# import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score # StratifiedKFold

from mne.decoding import CSP, Scaler
# from sklearn.decomposition import PCA
# import os
# from sklearn.svm import SVC
# from pyriemann.estimation import Covariances
# from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def pipe_setup(setting: str) -> Pipeline:
    if setting == "logreg":    
        return Pipeline([
                ('CSP', CSP(n_components=8, reg='ledoit_wolf',  log=True)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ])
    elif setting == "lda":
        return Pipeline([
                # ("scaler", Scaler(scalings="mean")),
                ('CSP', CSP(n_components=8, reg='ledoit_wolf',  log=True, norm_trace=True)),
                # ("scaler", StandardScaler()),
                ('LDA', LDA(solver='lsqr', shrinkage='auto'))
            ])
    elif setting == "mlp":
        return Pipeline([
                ('CSP', CSP(n_components=64, reg='ledoit_wolf',  log=True)),
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(64, 32),  
                    activation='relu',           
                    solver='adam',                
                    alpha=1e-4,                
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42
                ))
            ])
    else:
        raise ValueError("Wrong training pipeline")


def train(X, y, pipe_setting="logreg"):
    print(f"X shape: {X.shape}    y shape: {y.shape}")
    pipe = pipe_setup(pipe_setting)
    print(f"Training with cross validation. classifier: {pipe_setting}")
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"Training with no cross validation. classifier {pipe_setting}")
    pipe.fit(X, y)
    return pipe, scores.mean()
