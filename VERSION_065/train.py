import mne
import sys
from srcs import train, read_datafolder, split_dataset
import joblib

import warnings
warnings.filterwarnings("ignore")


def main():
    mne.set_log_level('WARNING')
    runs = [
        [ "R03", "R07", "R11"],
        [ "R04", "R08", "R12"],
        [ "R05", "R09", "R13"],
        [ "R06", "R10", "R14"],
    ]
    scores = []
    accs = []
    for i, run in enumerate(runs):
        print(f"Loading edf file loaded for current runs {i}:  {runs[i]}")
        subjects = read_datafolder(sys.argv[1], run)
        X, y, X_test, y_test = split_dataset(subjects)
        pipe, score = train(X, y, "lda")
        joblib.dump(pipe, f"ttv_{i}.pkl")
        scores.append(score)
        print("Testing")
        acc = pipe.score(X_test, y_test)
        accs.append(acc)
        print(f"Run {i} ---> final cross validation score: {score}   accuracy: {acc}\n")

    print()
    print("The final score of cross validation is: ", sum(scores) / len(scores))
    print("The final test accuracy is: ", sum(accs) / len(accs))


if __name__ == "__main__":
    main()
