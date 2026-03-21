from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score # StratifiedKFold

from mne.decoding import CSP, Scaler
from sklearn.svm import SVC

from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder


def pipe_setup(setting: str) -> Pipeline:
    if setting == "logreg":    
        return Pipeline([
                ('CSP', CSP(n_components=8, reg='ledoit_wolf',  log=True)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ])
    elif setting == "lda":
        return Pipeline([
                ('CSP', CSP(n_components=10, reg='ledoit_wolf', log=True, norm_trace=True)),
                ("scaler", StandardScaler()),
                ('LDA', LDA(solver='lsqr', shrinkage='auto'))
            ])
    elif setting == "svm":
        return Pipeline([
                ('CSP', CSP(n_components=10, reg='ledoit_wolf', log=True, norm_trace=True)),
                ("scaler", StandardScaler()),
                ('SVM', SVC(kernel='rbf', C=1.0, gamma='scale'))
            ])
    elif setting == "mlp":
        return Pipeline([
                ('CSP', CSP(n_components=64, reg='ledoit_wolf',  log=True)),
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(64, 32),  
                    activation='relu',           
                    solver='adam',                
                    alpha=1e-4,                
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42
                ))
            ])
    else:
        raise ValueError("Wrong training pipeline")


def train(X, y, pipe_setting="logreg"):
    print(f"X shape: {X.shape}    y shape: {y.shape}")
    le = None
    if pipe_setting == "xgb":
        le = LabelEncoder()
        y = le.fit_transform(y)
    pipe = pipe_setup(pipe_setting)
    print(f"Training with cross validation. classifier: {pipe_setting}")
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"Training with no cross validation. classifier {pipe_setting}")
    pipe.fit(X, y)
    return pipe, scores.mean()
