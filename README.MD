# Total-perspective-vortex

(Ongoing)

## Overview

This project is a machine learning task on EEG imaginary experiments prediction.

The project consists of 4 parts:
- EEG data processing
- Visualization of edf file
- Dimensionality reduction algorithm
- Classification train / predict


## Gettting started

The project is using python with mne, scikit-learn, to setup the environment:

```bash
make

source venv/bin/activate
```

To train and predict:

```bash
python train.py dataset
python predict.py -t 1 dataset/S089/S089R11.edf
# here 1 is the task index to perform, the task index has to match the task type
```

## Dataset

109 subjects have been tested on 4 tasks for 14 runs, with 64 electronode, capturing the EEG signals.
The subjects are asked to do the binary operation during each tasks:

- Task1 - open and close left or right fist
- Task2 - imagine opening and closing left or right fist
- Task3 - open and close both fists or both feet
- Task4 - imagine opening and closing both fists or both feet

And repeat all these tasks 3 times.

The experiments of 109 test subjects are store at .edf files, which can be read by mne library in python.

After training the model has to predict the event type the subject performs.

The minimum mean accuracy asked is 60 %, across all 4 experiments, on never seen data.


## EEG data

We use mne to read edf files, the raw input is noisy with distruptive brain activities.

Frequency filtering 8 - 30Hz to only keep the EEG signal frequency


## Dimensionality reduction

CSP algorithm is used for dimensionality reduction.


## Classification

We tested different machine learning classifiers in scikit-learn.

We also inplemented a customized logistic regression algorithm, compatible with sklearn pipeline object.

### Training

### 64 channels raw data:

|classifier | task1 | task2 | task3 | task4 | mean |
|-----------|-------|-------|-------|-------|------|
| CSP + LDA | 0.687(n=8) | 0.650(n=10) | 0.648(n=16) | 0.654(n=22) | 0.659 |
| CSP + LogisticRegression| 0.683(n=8) | 0.626(n=10) | 0.650(n=16) | 0.656(n=22) | 0.654 |
| CSP + XGBoost | 0.638(n=8) | 0.608(n=10) | 0.628(n=16) | 0.603(n=22) | 0.619 |
| CSP + SVM | 0.655(n=8) | 0.627(n=10) | 0.634(n=16) | 0.645(n=22) | 0.640 |
| CSP + MLP | 0.672(n=8) | 0.636(n=10) | 0.623(n=16) | 0.612(n=22) | 0.636 |


### 15 channels:

```python
motor_channels = [
    'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',
    'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',
    'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.'
]
```

|classifier | task1 | task2 | task3 | task4 | mean |
|-----------|-------|-------|-------|-------|------|
| CSP + LDA | 0.667 | 0.605 | 0.632 | 0.617 | 0.630 |
| CSP + LogisticRegression| 0.677 | 0.637 | 0.630 | 0.633 | 0.644 |
| CSP + SVM | 0.673 | 0.625 | 0.631 | 0.629 | 0.639 |
| CSP + mlp | nan | nan | nan | nan | nan |


### Prediction