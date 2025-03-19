#!/usr/bin/env python3
import os
import time
import logging
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/credit_score_data.csv"
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "credit_rating_xgb.json")

def build_preprocessor(numeric_features, categorical_features):
    # Builds a column transformer for scaling numeric and one-hot encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor

def plot_confusion_matrix_func(y_true, y_pred, classes):
    # Plots confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
    # Plots feature importance
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    if importance:
        df_importance = pd.DataFrame({
            "feature": list(importance.keys()),
            "importance": list(importance.values())
        }).sort_values(by="importance", ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=df_importance.head(20))
        plt.title("Feature Importance (Gain)")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        logging.info("No feature importance data available.")

def evaluate_model(model, X_test, y_test, feature_names):
    # Evaluates model and prints metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Poor", "Standard", "Good"]))

    plot_confusion_matrix_func(y_test, y_pred, classes=["Poor", "Standard", "Good"])

    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        logging.info(f"ROC AUC (ovr): {roc_auc:.4f}")
    except Exception:
        logging.warning("Failed to compute ROC AUC.")

    model.get_booster().feature_names = list(feature_names)
    plot_feature_importance(model, feature_names)

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.info(f"Created directory {MODEL_DIR}")

    data = pd.read_csv(DATA_PATH)
    X, y, numeric_features, categorical_features = process_data(data)

    if "Credit_Mix" in X.columns:
        X.drop("Credit_Mix", axis=1, inplace=True)
        if "Credit_Mix" in numeric_features:
            numeric_features.remove("Credit_Mix")
        if "Credit_Mix" in categorical_features:
            categorical_features.remove("Credit_Mix")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = numeric_features + categorical_features

    df_X_train = pd.DataFrame(X_train_proc, columns=feature_names)
    df_X_test = pd.DataFrame(X_test_proc, columns=feature_names)

    smote = SMOTE(random_state=42)
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    pipeline = ImbPipeline(steps=[
        ("smote", smote),
        ("clf", xgb_clf)
    ])

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 7],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    logging.info("Starting GridSearchCV on all processed features...")
    start_time = time.time()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=3
    )
    grid.fit(df_X_train, y_train)

    elapsed = time.time() - start_time
    logging.info(f"GridSearchCV completed in {elapsed:.2f} seconds")
    logging.info(f"Best parameters: {grid.best_params_}")

    best_model = grid.best_estimator_
    joblib.dump(preprocessor, PIPELINE_PATH)
    logging.info(f"Preprocessor saved at {PIPELINE_PATH}")

    best_model.named_steps["clf"].save_model(MODEL_PATH)
    logging.info(f"XGBoost model saved at {MODEL_PATH}")

    evaluate_model(
        model=best_model.named_steps["clf"],
        X_test=df_X_test,
        y_test=y_test,
        feature_names=feature_names
    )

if __name__ == "__main__":
    main()
