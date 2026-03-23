from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def pipe_setup(setting: str, n: int) -> Pipeline:
    if setting == "logreg":    
        return Pipeline([
                ("scaler", StandardScaler()),
                ('CSP', CSP(n_components=n, reg='ledoit_wolf',  log=True)),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
    elif setting == "lda":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True, norm_trace=True)),
                ('LDA', LDA(solver='lsqr', shrinkage='auto'))
            ])
    elif setting == "svm":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True, norm_trace=True)),
                ("scaler", StandardScaler()),
                ('SVM', SVC(kernel='rbf', C=1.0, gamma='scale'))
            ])
    elif setting == "mlp":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf',  log=True)),
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

    n_comps = [6, 7, 8, 9, 10, 12, 14, 15]

    max_score = 0
    max_pipe = pipe_setup(pipe_setting, n_comps[0])
    max_ncomp = 0
    
    for n_comp in n_comps:
        pipe = pipe_setup(pipe_setting, n_comp)
        print(f"Training with cross validation. classifier: {pipe_setting}")
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
