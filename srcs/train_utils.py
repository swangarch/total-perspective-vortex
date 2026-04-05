from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier

from .logreg import MyLogreg
from .CSP import MyCSP


def pipe_setup(setting: str, n: int) -> Pipeline:
    if setting == "CSP_Logreg":    
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=0.1)),
            ])
    elif setting == "CSP_MyLogreg":    
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True)),
                ("scaler", StandardScaler()),
                ("clf", MyLogreg(max_iter=1000, learning_rate=0.01, C=0.1)),
            ])
    elif setting == "MyCSP_MyLogreg":    
        return Pipeline([
                ('CSP', MyCSP(n_components=n)),
                ("scaler", StandardScaler()),
                ("clf", MyLogreg(max_iter=1000, learning_rate=0.01, C=0.1)),
            ])
    elif setting == "CSP_LDA":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True, norm_trace=True)),
                ('LDA', LDA(solver='lsqr', shrinkage='auto'))
            ])
    elif setting == "MyCSP_LDA":
        return Pipeline([
                ('CSP', MyCSP(n_components=n)),
                ('LDA', LDA(solver='lsqr', shrinkage='auto'))
            ])
    elif setting == "CSP_SVM":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True, norm_trace=True)),
                ("scaler", StandardScaler()),
                ('SVM', SVC(kernel='rbf', C=0.1, gamma='scale'))
            ])
    elif setting == "CSP_XGBOOST":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf', log=True, norm_trace=True)),
                ("scaler", StandardScaler()),
                ('XGBOOST', XGBClassifier(n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ))
            ])
    elif setting == "CSP_MLP":
        return Pipeline([
                ('CSP', CSP(n_components=n, reg='ledoit_wolf',  log=True)),
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(32, 16),  
                    activation='relu',           
                    solver='adam',                
                    alpha=1e-2,                
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True, 
                    n_iter_no_change=20, 
                ))
            ])
    else:
        raise ValueError("Wrong training pipeline")


def train(X, y, pipe_setting="logreg", n_comp: int = 8, search_param: bool = True):
    print(f"X shape: {X.shape}    y shape: {y.shape}")

    if search_param:
        n_comps = [6, 7, 8, 9, 10, 12, 14, 16]
    else:
        n_comps = [n_comp]
    pipes = [pipe_setting]
    
    max_score = -1
    max_pipe = pipe_setup(pipe_setting, n_comps[0])
    max_ncomp = 0
    
    for n_comp in n_comps:
        for p in pipes:
            pipe = pipe_setup(p, n_comp)
            print(f"Training with cross validation. classifier: {p}")
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
