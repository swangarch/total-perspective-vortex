# from mne.io import read_raw_edf
# import mne
import numpy as np
import os
import random as rd
# from .plot import show_single_epoch, show_edf
from .edf import read_edf


SEED = 2402


def read_datafolder(path: str, runs: list, task_index: int) -> tuple:
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
            res = read_edf(filepath, plot=False, task_index=task_index)
            sub_runs.append(res)
            count += 1
        subjects.append(sub_runs)
    print(f"All {count} edf file loaded for current runs.\n")
    return subjects


def create_dataset_arr(subjects: list, task_index) -> tuple:
    Xarr = []
    yarr = []
    read_count = 0
    drop_count = 0
    for sub in subjects:
        for res in sub:
            if res[0].ndim < 3 or len(res[0]) == 0:
                continue 
            if task_index in [1, 3, 4, 5]: # standarization
                mean = res[0].mean(axis=(0, 2), keepdims=True)
                std = res[0].std(axis=(0, 2), keepdims=True)
                data = (res[0] - mean) / std
            else:
                data = res[0]

            for i, epoch in enumerate(data):
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


def split_dataset(subjects: list, rate=0.8, task_index = 1):
    rd.seed(SEED)
    rd.shuffle(subjects)
    train_count = int(len(subjects) * rate)
    train_sub = subjects[:train_count]
    test_sub =  subjects[train_count:]
    print("Dataset has been splitted based on subjects")
    X, y = create_dataset_arr(train_sub, task_index)
    X_test, y_test = create_dataset_arr(test_sub, task_index)
    print("Dataset has been splitted to train set and test set.")
    return X, y, X_test, y_test


def preprocess_cross_subject_dataset(path, run, task_index) -> tuple:
    subjects = read_datafolder(path, run, task_index)
    return split_dataset(subjects, task_index=task_index)