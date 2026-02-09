from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from mne.decoding import CSP
import os


def train(X, y):
    pipe = Pipeline([
        ("csp", CSP(n_components=4, log=False)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=2500,
            random_state=42,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=0.001
        ))
    ])
    pipe.fit(X, y)
    
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    print("Train accuracy", acc)
    
    return pipe


def read_edf(file: str) -> np.ndarray:
    raw = read_raw_edf(file, preload=True)
    # print(type(raw))
    # events_from_annot, event_dict = mne.events_from_annotations(raw)
    # print(event_dict)
    # print(events_from_annot)
    # print(raw)
    # raw.filter(4, 14)
    raw.filter(4, 30)
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
    # print(epochs)
    # raw.plot()
    # plt.show()
    np_data = epochs.get_data()
    labels = epochs.events[:, -1]
    return np_data, labels


def read_datafolder(path: str) -> tuple:
    folders = os.listdir(path)

    runs = [ "R04", "R06", "R08", "R10", "R12", "R14" ]

    Xarr = []
    yarr = []
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        for file in files:
            if not file.endswith(".edf"): continue
            # print(file[-7:])
            if not file[-7:-4] in runs: continue
            filepath = os.path.join(path, folder, file)
            # print(filepath)
            res = read_edf(filepath)
            # print(res[0].shape)
            # print(res[1].shape)
            if res[0].shape[-1] == 641:
                Xarr.append(res[0])
                yarr.append(res[1])
        # break
    # for i, x in enumerate(Xarr):
    #     print(i, x.shape)
    Xarr_out = np.concatenate(Xarr, axis=0)
    yarr_out = np.concatenate(yarr, axis=0)
    # print(Xarr_out.shape)
    # print(yarr_out.shape)
    return Xarr_out, yarr_out


def main():
    X, y = read_datafolder(sys.argv[1])

    perm = np.random.permutation(len(X))

    split = int(0.8 * len(X))
    idx_X_train = perm[: split]
    idx_X_test = perm[split :]

    X_train = X[idx_X_train]
    X_test = X[idx_X_test]

    y_train = y[idx_X_train]
    y_test = y[idx_X_test]

    pipe = train(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy", acc)
    
    count = sum([1 for i in y if i == 2])
    print(count / len(y))
    # np_data, labels= read_edf(sys.argv[1])

    # print(np_data.shape)
    # print(labels.shape)
    # # chanel (61, 64, 160) epoch_nums, channels, signals
    # # data_channel1 = np_data[0][0]
    # # print(data_channel1.shape)
    # # plt.plot(data_channel1, lw=0.3)
    # # plt.show()

    # np_data2, labels2= read_edf(sys.argv[2])

    # pipe = train(np_data, labels)

    # y_pred2 = pipe.predict(np_data2)
    # acc2 = accuracy_score(labels2, y_pred2)
    # print(acc2)




if __name__ == "__main__":
    main()
