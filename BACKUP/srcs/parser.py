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
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
import os
import random as rd

# rd.seed(42)
rd.seed(2402)


def get_event(raw) -> dict:
    events, auto_event_id = mne.events_from_annotations(raw)

    event_id = None
    if "T1" in auto_event_id.keys() and "T2" in auto_event_id.keys():
        event_id = {
            "T1": auto_event_id["T1"],
            "T2": auto_event_id["T2"],
        }
    elif "T0" in auto_event_id.keys():
        event_id = {
            "T0": auto_event_id["T0"]
        }
    else:
        raise ValueError("Wrong EEG event type.")
    return events, event_id


def read_edf(file: str, plot: bool = False) -> np.ndarray:
    raw = read_raw_edf(file, preload=True, verbose=False)
    if plot:
        print(f"Visualize raw edf file: {file}")
        raw.plot()
        plt.show()
    raw.filter(8, 30)
    if plot:
        print(f"Visualize filtered raw edf file: {file}")
        raw.plot()
        plt.show()
    events, event_id = get_event(raw)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0,
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
    subjects = []
    count = 0
    for folder in folders: # each subject
        files = os.listdir(os.path.join(path, folder))
        sub_runs = []
        for file in files:
            if not file.endswith(".edf"): 
                continue
            if not file[-7:-4] in runs: 
                continue
            filepath = os.path.join(path, folder, file)
            res = read_edf(filepath)
            sub_runs.append(res)
            count += 1
        subjects.append(sub_runs)
    print(f"All {count} edf file loaded for current runs.\n")
    return subjects


def create_dataset_arr(subjects: list) -> tuple:
    Xarr = []
    yarr = []
    for sub in subjects:
        for res in sub:
            if res[0].shape[-1] == 641:
                Xarr.append(res[0])
                yarr.append(res[1])
    X = np.concatenate(Xarr, axis=0)
    y = np.concatenate(yarr, axis=0)
    return X, y


def split_dataset(subjects: list, rate=0.8):
    rd.shuffle(subjects)
    train_count = int(len(subjects) * rate)
    train_sub = subjects[:train_count]
    test_sub =  subjects[train_count:]
    print("Dataset has been splitted based on subjects")
    X, y = create_dataset_arr(train_sub)
    X_test, y_test = create_dataset_arr(test_sub)
    print("Dataset has been splitted to train set and test set.")
    return X, y, X_test, y_test