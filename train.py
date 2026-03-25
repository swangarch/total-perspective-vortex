import mne
import sys
from srcs import train, read_datafolder, split_dataset, read_data_subject, split_dataset_subject
import joblib
import os
import argparse

import warnings
warnings.filterwarnings("ignore")


def select_task(task: int) -> list:
    runs = [
        [ "R03", "R07", "R11"],
        [ "R04", "R08", "R12"],
        [ "R05", "R09", "R13"],
        [ "R06", "R10", "R14"],
    ]
    if task == 0:
        runs = runs
        print("Train model on all tasks")
    elif task in [1, 2, 3, 4]:
        runs = [runs[task - 1]]
        print(f"Train model on task {task}")
    else:
        raise ValueError("Wrong task ID")
    return runs


def run_train(path: str, model: str = "lda",
              search_param: bool = False,
              task: int = 1, subject: int = 0) -> None:
    mne.set_log_level('WARNING')
    runs = select_task(task)
    scores = []
    accs = []
    os.makedirs("models", exist_ok=True)
    for i, run in enumerate(runs):
        print(f"Loading edf file loaded for current runs {i}:  {runs[i]}")
        if subject == 0: # Cross subject tests
            print(f"Training across all subjects")
            subjects = read_datafolder(path, run)
            X, y, X_test, y_test = split_dataset(subjects)
        else:
            subject_path = os.path.join(path, "S" + str(subject).zfill(3))
            print(f"Training on single subject {subject}")
            Xarr, yarr = read_data_subject(subject_path, run)
            X, y, X_test, y_test = split_dataset_subject(Xarr, yarr)
        pipe, score = train(X, y, model, search_param=search_param)
        joblib.dump(pipe, f"models/ttv_{i}.pkl")
        scores.append(score)
        print(f"Testing on the X:{X_test.shape}  Y:{y_test.shape}")
        acc = pipe.score(X_test, y_test)
        accs.append(acc)
        print(f"Run {i} ---> final cross validation score: {score}   accuracy: {acc}\n")

    print()
    print("The final score of cross validation is: ", sum(scores) / len(scores))
    print("The final test accuracy is: ", sum(accs) / len(accs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder")
    parser.add_argument("--task", "-t", type=int, default=0)
    parser.add_argument("--subject", "-s", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="lda")
    parser.add_argument("--search_param", "-sm", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_train(args.datafolder, args.model, 
              args.search_param,
              args.task, args.subject)


if __name__ == "__main__":
    main()
