from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from mne.decoding import CSP


def train(X, y):
    pipe = Pipeline([
        ("csp", CSP(n_components=4, log=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=500,
            random_state=42,
            
        ))
    ])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    print(acc)
    return pipe


def read_edf(file: str) -> np.ndarray:
    raw = read_raw_edf(file, preload=True)
    print(type(raw))
    # events_from_annot, event_dict = mne.events_from_annotations(raw)
    # print(event_dict)
    # print(events_from_annot)
    # print(raw)
    raw.filter(4, 14)
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
    # print(epochs)
    # raw.plot()
    # plt.show()
    np_data = epochs.get_data()
    labels = epochs.events[:, -1]
    return np_data, labels


def main():
    
    np_data, labels= read_edf(sys.argv[1])

    print(np_data.shape)
    # chanel (61, 64, 160) epoch_nums, channels, signals
    # data_channel1 = np_data[0][0]
    # print(data_channel1.shape)
    # plt.plot(data_channel1, lw=0.3)
    # plt.show()

    np_data2, labels2= read_edf(sys.argv[2])
    pipe = train(np_data, labels)

    y_pred2 = pipe.predict(np_data2)
    acc2 = accuracy_score(labels2, y_pred2)
    print(acc2)




if __name__ == "__main__":
    main()
