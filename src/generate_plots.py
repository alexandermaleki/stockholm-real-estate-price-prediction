"""
Load the trained models and predictions, and regenerate the result
figures: predicted-vs-actual scatter plots, residual plots, the
train/test price distribution, and residual histograms.

Run preprocess.py and train_models.py first.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_artifacts():
    _, _, _, _, y_train, y_test = joblib.load("processed_data.pkl")
    baseline_df = joblib.load("baseline_df.pkl")
    best_xgb, xgb_pred = joblib.load("best_xgb.pkl")
    best_rf, rf_pred = joblib.load("best_rf.pkl")
    best_knn, knn_pred = joblib.load("best_knn.pkl")
    return y_train, y_test, baseline_df, xgb_pred, rf_pred, knn_pred


def build_comparison_table(y_test, baseline_df, xgb_pred, rf_pred, knn_pred):
    optimized_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "k-NN"],
        "MAE": [mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, xgb_pred), mean_absolute_error(y_test, knn_pred)],
        "RMSE": [np.sqrt(mean_squared_error(y_test, rf_pred)), np.sqrt(mean_squared_error(y_test, xgb_pred)), np.sqrt(mean_squared_error(y_test, knn_pred))],
        "R2": [r2_score(y_test, rf_pred), r2_score(y_test, xgb_pred), r2_score(y_test, knn_pred)],
    })
    optimized_df["Stage"] = "Optimized"

    baseline_df = baseline_df.rename(columns={"R²": "R2"})
    baseline_df["Stage"] = "Baseline"
    baseline_df = baseline_df[["Model", "MAE", "RMSE", "R2", "Stage"]]

    return pd.concat([baseline_df, optimized_df], ignore_index=True)


def plot_predicted_vs_actual(y_test, model_preds_by_filename):
    plt.style.use("default")
    y_test_msek = y_test / 1e6
    y_min, y_max = y_test_msek.min(), y_test_msek.max()
    tick_values = np.arange(5.0, 36.0, 5.0)

    for filename, y_pred in model_preds_by_filename.items():
        y_pred_msek = y_pred / 1e6
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(y_test_msek, y_pred_msek, alpha=0.5, s=30, color="#1f77b4")
        ax.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", linewidth=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Price (MSEK)")
        ax.set_ylabel("Predicted Price (MSEK)")
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=10)
        plt.tight_layout(pad=2.5)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_residuals(y_test, model_preds_named):
    y_test_msek = y_test / 1e6
    y_min, y_max = y_test_msek.min(), y_test_msek.max()
    tick_values = np.arange(5.0, 36.0, 5.0)

    for model_name, y_pred in model_preds_named.items():
        residuals_msek = (y_pred / 1e6) - y_test_msek
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(y_test_msek, residuals_msek, alpha=0.5, s=30, color="#1f77b4", label="Residuals")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual")
        ax.set_xlabel("Actual Price (MSEK)")
        ax.set_ylabel("Residual (Predicted - Actual) (MSEK)")
        ax.set_xlim(y_min, y_max)
        ax.set_xticks(tick_values)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.7), frameon=False, borderaxespad=0)
        plt.tight_layout()
        filename = model_name.lower().replace(" ", "_") + "_residuals.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_price_distribution(y_train, y_test):
    sns.set(style="white")
    plt.figure(figsize=(9, 5))
    sns.kdeplot(y_train, linewidth=2, label="Training Set", color="steelblue")
    sns.kdeplot(y_test, linewidth=2, label="Test Set", color="darkred")
    plt.xlabel("Apartment Price (SEK)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    sns.despine(top=True, right=True)
    plt.legend(title="", fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout()
    plt.savefig("price_distribution_kde.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_histograms(y_test, model_preds_named):
    all_residuals = []
    for y_pred in model_preds_named.values():
        all_residuals.extend(((y_pred - y_test) / 1e6).values)
    global_min, global_max = min(all_residuals), max(all_residuals)
    global_min *= 1.1 if global_min < 0 else 0.9
    global_max *= 1.1 if global_max > 0 else 0.9

    for model_name, y_pred in model_preds_named.items():
        residuals = (y_pred - y_test) / 1e6
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.hist(residuals, bins=40, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(global_min, global_max, 500)
        ax.plot(x, norm.pdf(x, mu, std), color="black", linewidth=2, label="Normal Distribution")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual")
        ax.set_xlim(global_min, global_max)
        ax.set_xlabel("Residual (Predicted - Actual) (MSEK)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.7), frameon=False, borderaxespad=0, fontsize=9)
        plt.tight_layout()
        filename = model_name.lower().replace(" ", "_") + "_residual_hist_msek.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    y_train, y_test, baseline_df, xgb_pred, rf_pred, knn_pred = load_artifacts()

    comparison_df = build_comparison_table(y_test, baseline_df, xgb_pred, rf_pred, knn_pred)
    print(comparison_df)

    model_preds_by_filename = {
        "random_forest_optimized.png": rf_pred,
        "xgboost_optimized.png": xgb_pred,
        "knn_optimized.png": knn_pred,
    }
    model_preds_named = {"Random Forest": rf_pred, "XGBoost": xgb_pred, "k-NN": knn_pred}

    plot_predicted_vs_actual(y_test, model_preds_by_filename)
    plot_residuals(y_test, model_preds_named)
    plot_price_distribution(y_train, y_test)
    plot_residual_histograms(y_test, model_preds_named)
    print("Saved all plots.")


if __name__ == "__main__":
    main()
