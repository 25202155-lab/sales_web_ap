import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score

def train_and_save():
    csv_file = "GROUP 7 [raw data file].csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Could not find '{csv_file}'. Make sure it is in the same directory."
        )

    print("Loading raw data...")
    df = pd.read_csv(csv_file)

    # Feature definitions based on optimal stepwise SAS output
    feature_cols = [
        "store_nbr",
        "family",
        "onpromotion",
        "oil_price",
        "month",
        "day_of_week",
        "day_of_month",
        "is_payday",
        "promo_flag",
    ]

    X_raw = df[feature_cols].copy()
    y_reg = df["sales"]
    y_clf = df["high_sales"]

    # Convert categorical variables into string categories for One-Hot Encoding
    X_raw["store_nbr"] = X_raw["store_nbr"].astype(str)
    X_raw["family"] = X_raw["family"].astype(str)

    # Perform One-Hot Encoding
    X_encoded = pd.get_dummies(
        X_raw, columns=["store_nbr", "family"], drop_first=False
    )
    feature_names = X_encoded.columns.tolist()

    # Train / Validation Split (70 / 30)
    X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = (
        train_test_split(
            X_encoded, y_reg, y_clf, test_size=0.3, random_state=42
        )
    )

    print("Training Regression Model (Stepwise equivalent)...")
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_reg_train)

    reg_pred = reg_model.predict(X_val)
    r2_val = r2_score(y_reg_val, reg_pred)
    ase_val = mean_squared_error(y_reg_val, reg_pred)
    print(f"Regression - Validation R2: {r2_val:.4f}, Validation ASE: {ase_val:.2f}")

    print("Training Random Forest Classification Model...")
    clf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    )
    clf_model.fit(X_train, y_clf_train)

    clf_pred = clf_model.predict(X_val)
    clf_prob = clf_model.predict_proba(X_val)[:, 1]
    acc_val = accuracy_score(y_clf_val, clf_pred)
    auc_val = roc_auc_score(y_clf_val, clf_prob)
    print(
        f"Classification - Validation Accuracy: {acc_val:.4f}, Area Under ROC: {auc_val:.4f}"
    )

    # Store model metadata, metadata levels, and feature names
    store_levels = sorted(df["store_nbr"].unique().tolist())
    family_levels = sorted(df["family"].unique().tolist())

    pipeline_data = {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "feature_names": feature_names,
        "store_levels": store_levels,
        "family_levels": family_levels,
        "metrics": {
            "reg_r2": r2_val,
            "reg_ase": ase_val,
            "clf_acc": acc_val,
            "clf_auc": auc_val,
        },
    }

    joblib.dump(pipeline_data, "model_pipeline.joblib")
    print("Successfully saved 'model_pipeline.joblib'!")

if __name__ == "__main__":
    train_and_save()
