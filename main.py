from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys


def main():
    raw = read_raw_edf(sys.argv[1])
    print(type(raw))

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    print(events_from_annot)


if __name__ == "__main__":
    main()