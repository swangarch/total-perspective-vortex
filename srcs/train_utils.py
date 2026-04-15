from sklearn.model_selection import cross_val_score
from .pipeline import pipe_setup
import mne
from .data import (
                  preprocess_cross_subject_dataset, 
                  preprocess_single_subject_dataset,
                  show_confusion_matrix
                )
from .task import select_label, select_task
import joblib
import os
from sklearn.metrics import classification_report


def train(X, y, pipe_setting="logreg", n_comp: int = 8, search_param: bool = True):
    print(f"X shape: {X.shape}    y shape: {y.shape}")

    if search_param:
        n_comps = [6, 7, 8, 9, 10, 12, 14, 16]
    else:
        n_comps = [n_comp]
    pipes = [pipe_setting]
    
    max_score = -1
    max_pipe = pipe_setup(pipe_setting, n_comps[0])
    max_ncomp = 0
    
    for n_comp in n_comps:
        for p in pipes:
            pipe = pipe_setup(p, n_comp)
            print(f"Training with cross validation. classifier: {p}")
            scores = cross_val_score(pipe, X, y, cv=5)
            curr_score = scores.mean()
            if curr_score > max_score:
                max_score = curr_score
                max_pipe = pipe
                max_ncomp = n_comp
            print(f"---- n_comps: {n_comp}  cross validation score: {scores.mean()}")

    print(f"Training with no cross validation. classifier {pipe_setting}  n_comps: {max_ncomp}")
    max_pipe.fit(X, y)
    return max_pipe, max_score


def run_train(path: str, config: dict = {},
              task: int = 1, subject: int = 0) -> None:
    mne.set_log_level('WARNING')
    runs = select_task(task)
    scores = []
    accs = []
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    for i, run in enumerate(runs):
        print(f"Edf file loaded for current runs {i}:  {runs[i]}")
        if task == 0:
            task_index = i + 1
        else:
            task_index = task
        task_conf = config["task"][str(task_index)]
        try:
            if subject == 0: # Cross subject tests
                print(f"Training across all subjects")
                X, y, X_test, y_test = preprocess_cross_subject_dataset(path, run, task_index)
            else:
                subject_path = os.path.join(path, "S" + str(subject).zfill(3))
                print(f"Training on single subject {subject}")
                X, y, X_test, y_test = preprocess_single_subject_dataset(subject_path, run, task_index)
        except Exception as e:
            print(f"Error: {e}")
            return
        pipe, score = train(X, y, task_conf["classifier"], n_comp=task_conf["n_comp"],
                            search_param=task_conf["search_param"])
        joblib.dump(pipe, f"models/ttv_{task_index}.pkl")
        scores.append(score)
        print(f"Testing on the X:{X_test.shape}  Y:{y_test.shape}")
        acc = pipe.score(X_test, y_test)
        labels = select_label(task_index)
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=labels[1:]))
        show_confusion_matrix(y_test, y_pred, labels, rf"results/Task{task_index}.png")
        accs.append(acc)
        print(f"Task {i + 1} ---> final cross validation score: {score}   accuracy: {acc}\n\n")

    print()
    final_score = sum(scores) / len(scores)
    final_acc = sum(accs) / len(accs)
    print("The final score of cross validation is: ", final_score)
    print(f"The final test accuracy is: ", final_acc)
