"""
Train Random Forest, XGBoost, and k-NN with time-aware cross-validation,
then tune each model's hyperparameters with Optuna (30 trials, 5-fold
time-series CV). Saves the trained models and their 2024 test-set
predictions for the plotting stage (generate_plots.py).

Run preprocess.py first to produce processed_data.pkl.
Requires: optuna-integration[sklearn], xgboost, scikit-learn
"""

import time
import joblib
import numpy as np
import pandas as pd
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, CategoricalDistribution
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

INPUT_FILE = "processed_data.pkl"
N_TRIALS = 30
N_JOBS = 2


def train_baseline_models(X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=N_JOBS),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=N_JOBS),
        "k-NN": KNeighborsRegressor(n_neighbors=5, n_jobs=N_JOBS),
    }

    # Time-aware cross-validation within the training set
    tscv = TimeSeriesSplit(n_splits=5)
    print("Time-Series Cross-Validation (2012-2023):\n")
    for name, model in models.items():
        fold = 1
        for train_idx, val_idx in tscv.split(X_train):
            if name == "k-NN":
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            else:
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            print(f"{name} | Fold {fold} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
            fold += 1

    print("\n--- Baseline Performance on 2024 Test Set ---")
    baseline_results = []
    for name, model in models.items():
        if name == "k-NN":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"{name} | MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        baseline_results.append({"Model": name, "Stage": "Baseline", "MAE": mae, "RMSE": rmse, "R²": r2})

    baseline_df = pd.DataFrame(baseline_results)
    return models, baseline_df


def tune_hyperparameters(X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test):
    cv = TimeSeriesSplit(n_splits=5)

    # --------- XGBoost ----------
    param_xgb = {
        "n_estimators": IntDistribution(100, 1000, step=100),
        "max_depth": IntDistribution(3, 7, step=2),
        "learning_rate": CategoricalDistribution([0.01, 0.05, 0.1]),
        "subsample": CategoricalDistribution([0.8, 1.0]),
        "colsample_bytree": CategoricalDistribution([0.8, 1.0]),
        "reg_lambda": CategoricalDistribution([0.1, 1]),
        "reg_alpha": CategoricalDistribution([0.1, 1]),
        "min_child_weight": CategoricalDistribution([1, 3]),
    }
    xgb_search = OptunaSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=N_JOBS),
        param_distributions=param_xgb, cv=cv, n_trials=N_TRIALS, scoring="r2", random_state=42,
    )
    start_time = time.time()
    xgb_search.fit(X_train, y_train)
    print(f"XGBoost tuning time: {time.time() - start_time:.2f}s")
    best_xgb = xgb_search.best_estimator_
    xgb_pred = best_xgb.predict(X_test)
    print("Optimized XGBoost:", xgb_search.best_params_)
    print(f"MAE: {mean_absolute_error(y_test, xgb_pred):.2f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}, "
          f"R2: {r2_score(y_test, xgb_pred):.2f}")

    # --------- Random Forest ----------
    param_rf = {
        "n_estimators": IntDistribution(100, 300, step=100),
        "max_depth": CategoricalDistribution([None, 10, 20, 30]),
        "min_samples_split": CategoricalDistribution([2, 5, 10]),
        "min_samples_leaf": CategoricalDistribution([1, 2, 4]),
    }
    rf_search = OptunaSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=N_JOBS),
        param_distributions=param_rf, cv=cv, n_trials=N_TRIALS, scoring="r2", random_state=42,
    )
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    print(f"Random Forest tuning time: {time.time() - start_time:.2f}s")
    best_rf = rf_search.best_estimator_
    rf_pred = best_rf.predict(X_test)
    print("Optimized Random Forest:", rf_search.best_params_)
    print(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}, "
          f"R2: {r2_score(y_test, rf_pred):.2f}")

    # --------- k-NN ----------
    param_knn = {
        "n_neighbors": CategoricalDistribution([3, 5, 7, 9, 11]),
        "weights": CategoricalDistribution(["uniform", "distance"]),
        "p": CategoricalDistribution([2]),
    }
    knn_search = OptunaSearchCV(
        estimator=KNeighborsRegressor(n_jobs=N_JOBS),
        param_distributions=param_knn, cv=cv, n_trials=N_TRIALS, scoring="r2", random_state=42,
    )
    start_time = time.time()
    knn_search.fit(X_train_scaled, y_train)
    print(f"k-NN tuning time: {time.time() - start_time:.2f}s")
    best_knn = knn_search.best_estimator_
    knn_pred = best_knn.predict(X_test_scaled)
    print("Optimized k-NN:", knn_search.best_params_)
    print(f"MAE: {mean_absolute_error(y_test, knn_pred):.2f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test, knn_pred)):.2f}, "
          f"R2: {r2_score(y_test, knn_pred):.2f}")

    return (best_xgb, xgb_pred), (best_rf, rf_pred), (best_knn, knn_pred)


def main():
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(INPUT_FILE)

    models, baseline_df = train_baseline_models(
        X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test
    )
    (best_xgb, xgb_pred), (best_rf, rf_pred), (best_knn, knn_pred) = tune_hyperparameters(
        X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test
    )

    joblib.dump(models, "baseline_models.pkl")
    joblib.dump(baseline_df, "baseline_df.pkl")
    joblib.dump((best_xgb, xgb_pred), "best_xgb.pkl")
    joblib.dump((best_rf, rf_pred), "best_rf.pkl")
    joblib.dump((best_knn, knn_pred), "best_knn.pkl")
    print("Saved trained models and predictions.")


if __name__ == "__main__":
    main()
