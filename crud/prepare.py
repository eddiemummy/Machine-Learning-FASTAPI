import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Optional
from schemas import Algorithms, regression_algorithms, classification_algorithms


def prepare_and_train(df: pd.DataFrame, algorithm_type: str, target: str, algorithm: Optional[str] = None):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm_type == 'CLASSIFICATION':
        algorithms = classification_algorithms
        if algorithm is None:
            test_scores = {}
            for alg_enum, alg_instance in algorithms.items():
                alg_instance.fit(X_train, y_train)
                preds = alg_instance.predict(X_test)
                test_scores[alg_enum.name] = round(accuracy_score(y_test, preds), 3)
            return test_scores
        else:
            model = algorithms[Algorithms[algorithm]]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return {algorithm: round(accuracy_score(y_test, pred), 3)}
    elif algorithm_type == 'REGRESSION':
        algorithms = regression_algorithms
        if algorithm is None:
            test_scores = {}
            for alg_enum, alg_instance in algorithms.items():
                alg_instance.fit(X_train, y_train)
                preds = alg_instance.predict(X_test)
                test_scores[alg_enum.name] = round(mean_squared_error(y_test, preds), 3)
            return test_scores
        else:
            model = algorithms[Algorithms[algorithm]]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return {algorithm: round(mean_squared_error(y_test, pred), 3)}

    else:
        raise ValueError("Invalid algorithm type. Expected 'CLASSIFICATION' or 'REGRESSION'.")
