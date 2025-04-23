# src/model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import pickle




def clean_and_transform_target(
    df: pd.DataFrame,
    target_col: str = 'Processing Time (Days)',
    log_transform: bool = True
) -> tuple[pd.DataFrame, str]:
    """Drops negative durations and optionally log1p-transforms the target."""
    df = df[df[target_col] >= 0].copy()
    if log_transform:
        new_col = target_col + '_log'
        df[new_col] = np.log1p(df[target_col])
        return df, new_col
    return df, target_col


def get_regressors():
    """Return a dict of model-name → estimator instance."""
    return {
        'Linear':        LinearRegression(),
        'Ridge':         Ridge(),
        'Lasso':         Lasso(),
        'DecisionTree':  DecisionTreeRegressor(random_state=42),
        'RandomForest':  RandomForestRegressor(random_state=42),
        'HistGB':        HistGradientBoostingRegressor(random_state=42),
        'SVR':           SVR(),
        'KNN':           KNeighborsRegressor(),
        'MLP':           MLPRegressor(random_state=42, max_iter=500),
    }

def evaluate_models(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Trains and evaluates each regressor, returning a DataFrame with MAE and R².
    If target_col ends with '_log', predictions and truths are expm1'd back.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # IMPUTE: median strategy
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    results = []
    for name, model in get_regressors().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # If log-transformed, invert back
        if target_col.endswith('_log'):
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y_test)
        else:
            y_true = y_test

        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)

        results.append({
            'Model': name,
            'MAE':   mae,
            'R2':    r2
        })

    return pd.DataFrame(results).sort_values('R2', ascending=False)


def save_model(model, filepath="saved_model.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath="saved_model.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)
