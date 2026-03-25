from mne.io import read_raw_edf
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import random as rd


rd.seed(2402)
np.random.seed(42)
# rd.seed(42)


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
    raw.pick("eeg")
    
    if plot:
        print(f"Visualize filtered raw edf file: {file}")
        raw.plot()
        plt.show()
    events, event_id = get_event(raw)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=-1,
        tmax=4,
        baseline=(-1, 0),
        preload=True,
        reject=dict(eeg=420e-6)
    )
    epochs.drop_bad()
    if len(epochs) == 0:                                 
        return np.array([]), np.array([])  
    epochs.crop(tmin=0, tmax=4)
    np_data = epochs.get_data()
    # labels = epochs.events[:, -1]
    raw_labels = epochs.events[:, -1]
    label_map = {v: i for i, v in enumerate(event_id.values())}
    labels = np.array([label_map[l] for l in raw_labels]) 
    # map label from 1 2 3 to 0 or 0, 1
    if plot:
        plt.title("Single epoch in 1 channel example.")
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.plot(np_data[0][1])
        plt.show()
    return np_data, labels


def read_datafolder(path: str, runs: list) -> tuple:
    folders = sorted(os.listdir(path))
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


def read_data_subject(path: str, runs: list) -> tuple:
    files = sorted(os.listdir(path))
    count = 0
    Xarr = []
    yarr = []
    for file in files: # each subject
        if not file.endswith(".edf"): 
            continue
        if not file[-7:-4] in runs:
            continue
        filepath = os.path.join(path, file)
        print(filepath)
        res = read_edf(filepath)
        if res[0].shape[-1] == 641:
            Xarr.append(res[0])
            yarr.append(res[1])
        else:
            print("\033[33mWarning: no standard length data is dropped.\033[0m")
        count += 1
    X = np.concatenate(Xarr, axis=0)
    y = np.concatenate(yarr, axis=0)
    print("X size of dataset", X.shape)
    print("y size of dataset", y.shape)

    print(f"T1 count: {np.sum(y==0)}   T2 count: {np.sum(y==1)}\n")
    
    print(f"All {count} edf file loaded for current runs.\n")
    return X, y


def create_dataset_arr(subjects: list) -> tuple:
    Xarr = []
    yarr = []
    read_count = 0
    drop_count = 0
    for sub in subjects:
        for res in sub:
            if res[0].shape[-1] == 641: #: #721:
                Xarr.append(res[0])
                yarr.append(res[1])
                read_count += 1
            else:
                print("\033[33mWarning: no standard length data is dropped.\033[0m")
                drop_count += 1
    print(f"{read_count} / {read_count + drop_count} data are used, {drop_count} are dropped")
    X = np.concatenate(Xarr, axis=0)
    y = np.concatenate(yarr, axis=0)
    
    print("X size of dataset", X.shape)
    print("y size of dataset", y.shape)
    print(f"T1 count: {np.sum(y==0)}   T2 count: {np.sum(y==1)}\n")
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


def split_dataset_subject(Xarr: np.array, yarr: np.array, rate=0.8):
    indices = np.random.permutation(len(Xarr))
    split = int(rate * len(Xarr))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X = Xarr[train_idx]
    X_test = Xarr[test_idx]

    y = yarr[train_idx]
    y_test = yarr[test_idx]

    print("Dataset has been splitted to train set and test set.")
    return X, y, X_test, y_test