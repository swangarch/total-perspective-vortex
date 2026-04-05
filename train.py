import mne
from srcs import (load_config, train, 
                  preprocess_cross_subject_dataset, 
                  preprocess_single_subject_dataset,
                  show_confusion_matrix,
                  select_label, select_task)
import joblib
import os
import argparse
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


def run_train(path: str, config: dict = {},
              task: int = 1, subject: int = 0) -> None:
    mne.set_log_level('WARNING')
    runs = select_task(task)
    scores = []
    accs = []
    os.makedirs("models", exist_ok=True)
    for i, run in enumerate(runs):
        print(f"Loading edf file loaded for current runs {i}:  {runs[i]}")
        if task == 0:
            task_index = i + 1
        else:
            task_index = task
        task_conf = config["task"][str(task_index)]
        try:
            if subject == 0: # Cross subject tests
                print(f"Training across all subjects")
                X, y, X_test, y_test = preprocess_cross_subject_dataset(path, run)
            else:
                subject_path = os.path.join(path, "S" + str(subject).zfill(3))
                print(f"Training on single subject {subject}")
                X, y, X_test, y_test = preprocess_single_subject_dataset(subject_path, run)
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
        show_confusion_matrix(y_test, y_pred, labels)
        accs.append(acc)
        print(f"Run {i} ---> final cross validation score: {score}   accuracy: {acc}\n\n")

    print()
    final_score = sum(scores) / len(scores)
    final_acc = sum(accs) / len(accs)
    print("The final score of cross validation is: ", final_score)
    print(f"The final test accuracy is: ", final_acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder")
    parser.add_argument("--task", "-t", type=int, default=0)
    parser.add_argument("--subject", "-s", type=int, default=0)
    parser.add_argument("--config", "-c", type=str, default="config.json")
    args = parser.parse_args()
    return args


def main():
    try:
        args = parse_args()
        config = load_config(args.config)
        run_train(args.datafolder, config,
              args.task, args.subject)
    except Exception as e:
        print("Error:", e)
    

if __name__ == "__main__":
    main()
