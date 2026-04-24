import numpy as np
import os
from .edf import read_edf


SEED = 2402


def read_data_single(filepath: str, task_index: int) -> tuple:  ## for test
    Xarr = []
    yarr = []
    if not filepath.endswith(".edf"): 
        return None
    res = read_edf(filepath, plot=False, task_index=task_index)
    for i in range(len(res[0])):
        if res[0].shape[-1] == 641:
            Xarr.append(res[0][i])
            yarr.append(res[1][i])
        else:
            print(res[i].shape[-1])
            print("\033[33mWarning: no standard length data is dropped.\033[0m")
    X = np.stack(Xarr, axis=0)
    y = np.stack(yarr, axis=0)
    print("X shape", X.shape, "  ", "y shape", y.shape)
    print(f"T1 count: {np.sum(y==0)}   T2 count: {np.sum(y==1)}\n")
    print(f"Edf file {filepath} loaded.\n")
    return X, y


def read_data_subject(path: str, runs: list, task_index: int = 2) -> tuple:
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
        res = read_edf(filepath, plot=False, task_index=task_index)
        read_res.append(res)
        count += 1
        
    print(f"All {count} edf file loaded for current runs.\n")

    for res in read_res:
        if res[0].ndim < 3 or len(res[0]) == 0:                       
          continue
        mean = res[0].mean(axis=(0, 2), keepdims=True)
        std = res[0].std(axis=(0, 2), keepdims=True)
        res[0] = (res[0] - mean) / std

    X_train_arr = []
    y_train_arr = []

    X_test_arr = []
    y_test_arr = []
    for i, res in enumerate(read_res):
        for j, epoch in enumerate(res[0]):
            if epoch.shape[-1] == 641:
                if (task_index in [1, 2, 3, 4, 5] and i != 2 ) or (task_index in [5, 6] and i not in [4, 5]):
                    X_train_arr.append(epoch)
                    y_train_arr.append(res[1][j])
                else:
                    X_test_arr.append(epoch)
                    y_test_arr.append(res[1][j])
            else:
                print(f"\033[33mWarning: no standard length data is dropped, shape {epoch.shape[-1]}.\033[0m")

    X_train = np.stack(X_train_arr, axis=0)
    y_train = np.array(y_train_arr)
    print("Train data X shape", X_train.shape, "  ", "y shape", y_train.shape)

    X_test = np.stack(X_test_arr, axis=0)
    y_test = np.array(y_test_arr)
    print("Test data X shape", X_test.shape, "  ", "y shape", y_test.shape)

    return X_train, y_train, X_test, y_test


def preprocess_single_subject_dataset(path: str, run: list, test_idx: int = 2) -> tuple:
    return read_data_subject(path, run, test_idx)
