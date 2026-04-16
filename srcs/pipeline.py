from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
