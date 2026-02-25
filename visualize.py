from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from mne.decoding import CSP
# import os


def read_edf(file: str) -> np.ndarray:
    raw = read_raw_edf(file, preload=True)
    # print("RAW DATA IS", raw)
    raw.plot()
    # plt.title("Raw data")
    plt.show()
    raw.filter(8, 14)
    raw.plot()
    # plt.title("Filtered data")
    plt.show()
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
    np_data = epochs.get_data()
    labels = epochs.events[:, -1]
    # print(np_data[1])

    plt.plot(np_data[0][1])
    plt.show()

    return np_data, labels


def main():
    read_edf(sys.argv[1])


if __name__ == "__main__":
    main()
