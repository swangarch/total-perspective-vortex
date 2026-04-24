import joblib
from sklearn.pipeline import Pipeline
import argparse
from srcs import (
                    load_config, 
                    read_data_single, 
                    show_confusion_matrix
                )
from sklearn.metrics import classification_report
import mne
import warnings
import time
from datetime import datetime
warnings.filterwarnings("ignore")


def load_model(path: str) -> Pipeline:
    model = joblib.load(path)
    return model


def run_predict(path: str, config: dict = {},
              task: int = 1) -> None:
    mne.set_log_level('WARNING')
    conf = config["task"][str(task)]
    labels = conf["labels"]
    print(conf["title"])
    model_file = conf["model"]
    pipe = load_model(model_file)
    print(pipe, "\n")
    X_test, y_test = read_data_single(path, task)
    print(f"Realtime stream data testing on the X:{X_test.shape}  Y:{y_test.shape}")
    acc = pipe.score(X_test, y_test)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))
    show_confusion_matrix(y_test, y_pred, conf["title"], conf["labels"], rf"results/Task{task}.png")
    print(f"Run {path} ---> final accuracy: {acc}\n\n")
    print(f"Prediction ---> {y_pred}")
    print(f"Ground Truth: ---> {y_test}")
    print()


def run_predict_realtime(path: str, config: dict = {},
              task: int = 1) -> None:
    mne.set_log_level('WARNING')
    
    conf = config["task"][str(task)]
    labels = conf["labels"]
    print(conf["title"])
    model_file = conf["model"]
    pipe = load_model(model_file)
    X_test, y_test = read_data_single(path, task)
    print(f"Testing on the X:{X_test.shape}  Y:{y_test.shape}")
    correct_count = 0
    time.sleep(4)
    for i in range(len(X_test)):
        if i == 0:
            time.sleep(4)
        else:
            time.sleep(8)
        y_pred = pipe.predict(X_test[i: i + 1])
        is_correct = (y_test[i] == y_pred)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   Epoch {i}  Ground Truth: {y_test[i]}  Prediction: {y_pred[0]}   Correct prediction: {is_correct}")
        
        print(f"---- Ground Truth: {labels[y_test[i]]}")
        print(f"---- Prediction: {labels[y_pred[0]]}")
        if is_correct:
            correct_count += 1
    print(f"Final accuracy: {float(correct_count) / float(len(X_test))}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("--task", "-t", type=int, default=1)
    parser.add_argument("--config", "-c", type=str, default="config.json")
    parser.add_argument("--realtime", "-r", action="store_true")
    args = parser.parse_args()
    assert args.task in [0, 1, 2, 3, 4, 5, 6], "Wrong task index"
    return args


def main():
    try:
        args = parse_args()
        config = load_config(args.config)
        if args.realtime:
            run_predict_realtime(args.datapath, config, args.task)
        else:
            run_predict(args.datapath, config, args.task)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
