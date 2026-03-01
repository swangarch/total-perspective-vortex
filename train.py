import mne
import sys
from srcs import train, read_datafolder


def main():
    mne.set_log_level('WARNING')
    runs = [
        [ "R03", "R07", "R11"],
        # [ "R04", "R08", "R12"],
        # [ "R05", "R09", "R13"],
        # [ "R06", "R10", "R14"],
    ]
    scores = []
    for run in runs:
        X, y = read_datafolder(sys.argv[1], run)
        pipe, score = train(X, y)
        scores.append(score)
    print("The final score is: ", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
