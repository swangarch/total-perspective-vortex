from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import sys
import numpy as np

from srcs import read_edf


def main():
    try:
        mne.set_log_level('WARNING')
        if len(sys.argv) != 2:
            print("Error: Wrong argument number. Please enter edf file as argument.")
            return    
        read_edf(sys.argv[1], plot=True)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
