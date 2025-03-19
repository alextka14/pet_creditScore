import os
import logging
import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "credit_rating_xgb.json")

def load_pipeline():
    try:
        preprocessor = joblib.load(PIPELINE_PATH)
        logging.info("Pipeline loaded successfully.")
        return preprocessor
    except Exception as e:
        logging.error(f"Error loading pipeline: {e}")
        raise

def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def prepare_predict_data(df):
    # Prepares data similarly to the process in train.py
    drop_columns = ["ID", "Customer_ID", "SSN", "Name"]
    if "Credit_Score" in df.columns:
        df = df.drop(columns=["Credit_Score"], errors="ignore")

    if 'Type_of_Loan' in df.columns:
        df['Type_of_Loan'] = df['Type_of_Loan'].fillna("").apply(
            lambda x: [item.strip() for item in x.split(',') if item.strip()]
        )
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        loan_type_df = pd.DataFrame(
            mlb.fit_transform(df.pop('Type_of_Loan')),
            columns=[f"LoanType_{c}" for c in mlb.classes_],
            index=df.index
        )
        df = pd.concat([df, loan_type_df], axis=1)

    df.drop(columns=drop_columns, errors="ignore", inplace=True)
    return df

def main(input_csv):
    preprocessor = load_pipeline()
    model = load_model()
    data = pd.read_csv(input_csv)

    y_true = None
    if "Credit_Score" in data.columns:
        data['Credit_Score'] = data['Credit_Score'].map({"Poor": 0, "Standard": 1, "Good": 2})
        y_true = data["Credit_Score"]

    X = prepare_predict_data(data)
    X_processed = preprocessor.transform(X)
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)

    if y_true is not None:
        accuracy = accuracy_score(y_true, y_pred)
        logging.info(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Poor", "Standard", "Good"]))

        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            logging.info(f"ROC AUC (ovr): {roc_auc:.4f}")
        except Exception:
            logging.warning("Failed to compute ROC AUC.")

        plot_confusion_matrix(y_true, y_pred, classes=["Poor", "Standard", "Good"])

    mapping = {0: "Poor", 1: "Standard", 2: "Good"}
    results = pd.DataFrame({
        "Predicted": [mapping[i] for i in y_pred],
        "Prob_Poor": y_proba[:, 0],
        "Prob_Standard": y_proba[:, 1],
        "Prob_Good": y_proba[:, 2]
    })

    if y_true is not None:
        results["Actual"] = [mapping[i] for i in y_true]

    print("\nSample Predictions (first 10):")
    print(results.head(10))

    if len(X) > 0:
        example_idx = 0
        print(f"\nDetails for example at index {example_idx}:")
        print("Original features after prepare_predict_data:")
        print(X.iloc[example_idx])
        if y_true is not None:
            print("Actual class:", mapping[y_true.iloc[example_idx]])
        print("Predicted class:", mapping[y_pred[example_idx]])
        print("Probabilities:", y_proba[example_idx])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Scoring Model Inference Script")
    parser.add_argument("input_csv", type=str, help="Path to CSV file with test data (with 'Credit_Score' column if available)")
    args = parser.parse_args()
    main(args.input_csv)
