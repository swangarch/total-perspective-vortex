from mne.io import read_raw_edf
# impo
import mne
import numpy as np
import os
import random as rd
from .plot import show_single_epoch, show_edf


SEED = 2402


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


def get_run_id(file: str) -> int:
    return int(file[-6:-4])


def handle_do_imagine(np_data: np.array, labels: np.array, 
                      file: str, task_index: int) -> tuple:
    do = [3, 7, 11]
    imagine = [4, 8, 12]
    run_id = get_run_id(file)
    if task_index == 5:
        if run_id in do:
            np_data = np_data[labels == 0]
            labels = labels[labels == 0]
        elif run_id in imagine:
            np_data = np_data[labels == 0]
            labels = labels[labels == 0] + 1
    elif task_index == 6:
        if run_id in do:
            np_data = np_data[labels == 1]
            labels = labels[labels == 1] - 1
        elif run_id in imagine:
            np_data = np_data[labels == 1]
            labels = labels[labels == 1]
    return [np_data, labels]


def read_edf(file: str, plot: bool = False, task_index: int = 0) -> np.ndarray:

    raw = read_raw_edf(file, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    if plot:
        show_edf(raw, montage, file, "raw edf")

    raw.pick("eeg")
    raw.notch_filter(60)
    raw.filter(8, 30)
    raw.resample(160)
    
    if plot:
        show_edf(raw, montage, file, "filtered raw edf")

    events, event_id = get_event(raw)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=-1,
        tmax=4,
        baseline=(-1, 0),
        preload=True,
        # reject=dict(eeg=320e-6),
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
    # if plot:
    #     show_single_epoch(np_data)
    
    return handle_do_imagine(np_data, labels, file, task_index)
