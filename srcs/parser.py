from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from mne.decoding import CSP
import os


def get_event(raw) -> dict:
    events, auto_event_id = mne.events_from_annotations(raw)

    event_id = None
    if "T1" in auto_event_id.keys() and "T2" in auto_event_id.keys():
        event_id = {
            "T1": 2,
            "T2": 3,
        }
    elif "T0" in auto_event_id.keys():
        event_id = {
            "T0": 1
        }
    else:
        raise ValueError("Wrong EEG event type.")
    return event_id


def read_edf(file: str, plot: bool = False) -> np.ndarray:
    raw = read_raw_edf(file, preload=True)
    if plot:
        print(f"Visualize raw edf file: {file}")
        raw.plot()
        plt.show()
    # raw.filter(8, 14)
    raw.filter(8, 14)
    if plot:
        print(f"Visualize filtered raw edf file: {file}")
        raw.plot()
        plt.show()
    # event_id = get_event(raw)
    epochs = mne.Epochs(
        raw,
        event_id=get_event(raw),
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True
    )
    epochs.drop_bad()
    np_data = epochs.get_data()
    labels = epochs.events[:, -1]

    if plot:
        plt.title("Single epoch in 1 channel example.")
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.plot(np_data[0][1])
        plt.show()

    return np_data, labels


def read_datafolder(path: str, runs: list) -> tuple:
    folders = os.listdir(path)

    Xarr = []
    yarr = []
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        for file in files:
            if not file.endswith(".edf"): continue
            if not file[-7:-4] in runs: continue
            filepath = os.path.join(path, folder, file)
            res = read_edf(filepath)
            print(f"EDF file: {filepath} has been loaded.")
            if res[0].shape[-1] == 641:
                Xarr.append(res[0])
                yarr.append(res[1])
    Xarr_out = np.concatenate(Xarr, axis=0)
    yarr_out = np.concatenate(yarr, axis=0)
    return Xarr_out, yarr_out
