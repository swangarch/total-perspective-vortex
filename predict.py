import joblib
from sklearn.pipeline import Pipeline


def load_model(path: str) -> Pipeline:
    model = joblib.load(path)
    return model


def predict(model, X_test, y_test):
    return model.predict(X_test), model.score(X_test, y_test)


def main():
    pass


if __name__ == "__main__":
    main()