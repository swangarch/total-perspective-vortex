from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from mne.decoding import CSP
import os


def train(X, y):
    pipe = Pipeline([
        ("csp", CSP(n_components=64, log=True)),
        # ("lda", LinearDiscriminantAnalysis())
        # ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 16, 8),
            activation="relu",
            max_iter=1500,
            solver="adam",
            random_state=42,
            batch_size=16,
            alpha=0.001,
            tol=0,
            learning_rate="constant",
            learning_rate_init=0.0005,
            early_stopping=False,
            shuffle=True
            # validation_fraction=0.15,
        ))
    ])
    pipe.fit(X, y)

    mlp = pipe.named_steps["mlp"]
    plt.plot(mlp.loss_curve_)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # print(mlp.cr)

    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    print("Train accuracy", acc)

    scores = cross_val_score(pipe, X, y, cv=5)
    print("Train score", scores.mean())
    # print()
    # print()

    return pipe


def read_edf(file: str) -> np.ndarray:
    raw = read_raw_edf(file, preload=True)
    raw.filter(8, 14)
    event_id = {
        "T1": 2,
        "T2": 3,
    }
    epochs = mne.Epochs(
        raw,
        event_id=event_id,
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True
    )
    epochs.drop_bad()
    np_data = epochs.get_data()
    labels = epochs.events[:, -1]
    return np_data, labels


def read_datafolder(path: str) -> tuple:
    folders = os.listdir(path)

    runs = [ "R03", "R07", "R11"]
    runs = [ "R04", "R08", "R12"]
    runs = [ "R05", "R09", "R13"]
    # runs = [ "R06", "R10", "R14"]

    Xarr = []
    yarr = []
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        for file in files:
            if not file.endswith(".edf"): continue
            if not file[-7:-4] in runs: continue
            filepath = os.path.join(path, folder, file)
            res = read_edf(filepath)
            if res[0].shape[-1] == 641:
                Xarr.append(res[0])
                yarr.append(res[1])
    Xarr_out = np.concatenate(Xarr, axis=0)
    yarr_out = np.concatenate(yarr, axis=0)
    return Xarr_out, yarr_out


def split_data(X, y):
    perm = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    idx_X_train = perm[: split]
    idx_X_test = perm[split :]
    X_train = X[idx_X_train]
    X_test = X[idx_X_test]
    y_train = y[idx_X_train]
    y_test = y[idx_X_test]
    return X_train, X_test, y_train, y_test


def predict(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy", acc)

    scores = cross_val_score(pipe, X_test, y_test, cv=5)

    print("Test score", scores.mean())
    return y_pred


def main():
    X, y = read_datafolder(sys.argv[1])
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = train(X_train, y_train)
    predict(pipe, X_test, y_test)


if __name__ == "__main__":
    main()
