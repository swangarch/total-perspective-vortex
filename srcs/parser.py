from mne.io import read_raw_edf
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import random as rd
from .plot import show_single_epoch, show_edf


SEED = 2402 # 0.6599
# SEED = 24921 # 0.658


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
        show_edf(raw)
    raw.resample(160)
    raw.filter(8, 30)
    raw.pick("eeg")
    # motor_channels = [
    #     'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',  # central
    #     'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',  # front
    #     'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.'   # back
    # ]
    # raw.pick(motor_channels)
    if plot:
        print(f"Visualize filtered raw edf file: {file}")
        show_edf(raw)
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
    raw_labels = epochs.events[:, -1]
    label_map = {v: i for i, v in enumerate(event_id.values())}
    labels = np.array([label_map[l] for l in raw_labels]) 
    # map label from 1 2 3 to 0 or 0, 1
    if plot:
        show_single_epoch(np_data)
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


def create_dataset_arr(subjects: list) -> tuple:
    Xarr = []
    yarr = []
    read_count = 0
    drop_count = 0
    for sub in subjects:
        for res in sub:
            for i, epoch in enumerate(res[0]):
                if epoch.shape[-1] == 641: #641: #: #721:
                    Xarr.append(epoch)
                    yarr.append(res[1][i])
                    read_count += 1
                else:
                    print(f"\033[33mWarning: no standard length data is dropped. Length {epoch.shape[-1]}\033[0m")
                    drop_count += 1
    print(f"{read_count} / {read_count + drop_count} data are used, {drop_count} are dropped")
    X = np.stack(Xarr, axis=0)
    y = np.array(yarr)
    print("X shape:", X.shape, "  ", "y shape:", y.shape)
    print(f"T1 count: {np.sum(y==0)}   T2 count: {np.sum(y==1)}\n")
    return X, y


def split_dataset(subjects: list, rate=0.8):
    rd.seed(SEED)
    rd.shuffle(subjects)
    train_count = int(len(subjects) * rate)
    train_sub = subjects[:train_count]
    test_sub =  subjects[train_count:]
    print("Dataset has been splitted based on subjects")
    X, y = create_dataset_arr(train_sub)
    X_test, y_test = create_dataset_arr(test_sub)
    print("Dataset has been splitted to train set and test set.")
    return X, y, X_test, y_test


def preprocess_cross_subject_dataset(path, run) -> tuple:
    subjects = read_datafolder(path, run)
    return split_dataset(subjects)


## ------------------for single subject----------------------


# def read_data_subject(path: str, runs: list) -> tuple:
#     files = sorted(os.listdir(path))
#     count = 0
#     Xarr = []
#     yarr = []
#     for file in files: # each subject
#         if not file.endswith(".edf"): 
#             continue
#         if not file[-7:-4] in runs:
#             continue
#         filepath = os.path.join(path, file)
#         print(filepath)
#         res = read_edf(filepath)
#         if res[0].shape[-1] == 641:
#             Xarr.append(res[0])
#             yarr.append(res[1])
#         else:
#             print("\033[33mWarning: no standard length data is dropped.\033[0m")
#         count += 1
#     X = np.concatenate(Xarr, axis=0)
#     y = np.concatenate(yarr, axis=0)
#     print("X size of dataset", X.shape)
#     print("y size of dataset", y.shape)

#     print(f"T1 count: {np.sum(y==0)}   T2 count: {np.sum(y==1)}\n")
    
#     print(f"All {count} edf file loaded for current runs.\n")
#     return X, y


# def split_dataset_subject(Xarr: np.array, yarr: np.array, rate=0.8):
#     rd.seed(SEED)
#     indices = np.random.permutation(len(Xarr))
#     split = int(rate * len(Xarr))
#     train_idx = indices[:split]
#     test_idx = indices[split:]

#     X = Xarr[train_idx]
#     X_test = Xarr[test_idx]

#     y = yarr[train_idx]
#     y_test = yarr[test_idx]

#     print("Dataset has been splitted to train set and test set.")
#     return X, y, X_test, y_test


# def preprocess_single_subject_dataset(path, run) -> tuple:
#     X, y = read_data_subject(path, run)
#     return split_dataset_subject(X, y)


def read_data_subject(path: str, runs: list, test_idx: int = 2) -> tuple:
    files = sorted(os.listdir(path))
    count = 0

    read_res = []
    for file in files: # each subject
        if not file.endswith(".edf"): 
            continue
        if not file[-7:-4] in runs:
            continue
        filepath = os.path.join(path, file)
        print(filepath)
        res = read_edf(filepath)
        read_res.append(res)
        count += 1
        
    print(f"All {count} edf file loaded for current runs.\n")

    X_train_arr = []
    y_train_arr = []

    X_test_arr = []
    y_test_arr = []
    for i, res in enumerate(read_res):
        for j, epoch in enumerate(res[0]):
            if epoch.shape[-1] == 641:
                if i != test_idx:
                    X_train_arr.append(epoch)
                    y_train_arr.append(res[1][j])
                else:
                    X_test_arr.append(epoch)
                    y_test_arr.append(res[1][j])
            else:
                print(f"\033[33mWarning: no standard length data is dropped, shape {epoch.shape[-1]}.\033[0m")

    X_train = np.stack(X_train_arr, axis=0)
    y_train = np.array(y_train_arr)
    print("X shape", X_train.shape, "  ", "y shape", y_train.shape)

    X_test = np.stack(X_test_arr, axis=0)
    y_test = np.array(y_test_arr)
    print("X shape", X_test.shape, "  ", "y shape", y_test.shape)

    return X_train, y_train, X_test, y_test


def preprocess_single_subject_dataset(path, run) -> tuple:
    return read_data_subject(path, run)