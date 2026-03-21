"""Test different channel pick strategies vs no-pick baseline."""
import mne
import numpy as np
import random as rd
import os
import sys
from mne.io import read_raw_edf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')

rd.seed(2402)

PICK_CONFIGS = {
    "no_pick (all 64)": None,
    "motor_core (3ch)": ['C3..', 'C4..', 'Cz..'],
    "motor_7ch": ['C3..', 'C4..', 'Cz..', 'Fc3.', 'Fc4.', 'Cp3.', 'Cp4.'],
    "motor_13ch": ['C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
                   'Fc3.', 'Fcz.', 'Fc4.', 'Cp3.', 'Cpz.', 'Cp4.'],
    "motor_21ch": ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
                   'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
                   'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.'],
}


def get_event(raw):
    events, auto_event_id = mne.events_from_annotations(raw)
    if "T1" in auto_event_id and "T2" in auto_event_id:
        event_id = {"T1": auto_event_id["T1"], "T2": auto_event_id["T2"]}
    elif "T0" in auto_event_id:
        event_id = {"T0": auto_event_id["T0"]}
    else:
        raise ValueError("Wrong EEG event type.")
    return events, event_id


def load_data(path, runs, pick_channels=None):
    folders = sorted(os.listdir(path))
    subjects = []
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        sub_runs = []
        for file in files:
            if not file.endswith(".edf"):
                continue
            if file[-7:-4] not in runs:
                continue
            filepath = os.path.join(path, folder, file)
            raw = read_raw_edf(filepath, preload=True, verbose=False)
            raw.filter(8, 30)
            if pick_channels is not None:
                raw.pick(pick_channels)
            events, event_id = get_event(raw)
            epochs = mne.Epochs(raw, events=events, event_id=event_id,
                                tmin=0.5, tmax=4.0, baseline=None,
                                preload=True, reject=dict(eeg=120e-6))
            epochs.drop_bad()
            np_data = epochs.get_data()
            labels = epochs.events[:, -1]
            if np_data.shape[0] > 0 and np_data.shape[-1] == 561:
                sub_runs.append((np_data, labels))
        subjects.append(sub_runs)
    return subjects


def build_dataset(subjects, rate=0.8):
    subjects = [s for s in subjects if len(s) > 0]
    rd.shuffle(subjects)
    split = int(len(subjects) * rate)
    def concat(subs):
        arrays = [r[0] for s in subs for r in s]
        labels = [r[1] for s in subs for r in s]
        if not arrays:
            return np.array([]), np.array([])
        return np.concatenate(arrays, axis=0), np.concatenate(labels, axis=0)
    X, y = concat(subjects[:split])
    Xt, yt = concat(subjects[split:])
    return X, y, Xt, yt


def run_test(path, runs, pick_channels=None):
    subjects = load_data(path, runs, pick_channels)
    subjects = [s for s in subjects if len(s) > 0]
    arrays = [r[0] for s in subjects for r in s]
    labels = [r[1] for s in subjects for r in s]
    if not arrays:
        return 0, 0, 0.0
    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    n_ch = X.shape[1]
    n_comp = min(10, n_ch)
    pipe = Pipeline([
        ('CSP', CSP(n_components=n_comp, reg='ledoit_wolf', log=True, norm_trace=True)),
        ('scaler', StandardScaler()),
        ('LDA', LDA(solver='lsqr', shrinkage='auto'))
    ])
    print(f"  samples: {X.shape[0]}, channels: {n_ch}")
    cv_folds = min(5, X.shape[0] // 4)
    if cv_folds < 2:
        return n_ch, n_comp, 0.0
    cv_score = cross_val_score(pipe, X, y, cv=cv_folds).mean()
    return n_ch, n_comp, cv_score


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_example"
    runs = ["R04", "R08", "R12"]

    print(f"{'Config':<22} {'#Ch':>4} {'CSP':>4} {'CV Score':>9}")
    print("-" * 45)

    for name, channels in PICK_CONFIGS.items():
        rd.seed(2402)
        n_ch, n_comp, cv = run_test(data_path, runs, channels)
        print(f"{name:<22} {n_ch:>4} {n_comp:>4} {cv:>9.4f}")
