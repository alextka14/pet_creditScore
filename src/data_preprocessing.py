import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

DROP_COLUMNS = ["ID", "Customer_ID", "SSN", "Name"]


def process_type_of_loan(data):
    """
    Process the 'Type_of_Loan' column if present.
    Splits the string by commas and strips spaces,
    then applies MultiLabelBinarizer.
    """
    if 'Type_of_Loan' in data.columns:
        data['Type_of_Loan'] = data['Type_of_Loan'].fillna("").apply(
            lambda x: [item.strip() for item in x.split(',') if item.strip()]
        )
        mlb = MultiLabelBinarizer()
        loan_type_df = pd.DataFrame(
            mlb.fit_transform(data.pop('Type_of_Loan')),
            columns=[f"LoanType_{c}" for c in mlb.classes_],
            index=data.index
        )
        return loan_type_df
    return pd.DataFrame(index=data.index)


def process_data(data):
    """
    Process training data:
    - Map Credit_Score to numeric values (Poor=0, Standard=1, Good=2)
    - Process the 'Type_of_Loan' column if present
    - Drop unnecessary columns
    - Split into features (X) and target (y)

    Returns: X, y, numeric_features, categorical_features
    """
    target_col = "Credit_Score"
    drop_columns = DROP_COLUMNS.copy()

    data[target_col] = data[target_col].map({"Poor": 0, "Standard": 1, "Good": 2})

    loan_type_df = process_type_of_loan(data)

    data.drop(columns=drop_columns, errors="ignore", inplace=True)

    y = data[target_col]
    X = data.drop(columns=[target_col], errors="ignore")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    if not loan_type_df.empty:
        X = pd.concat([X, loan_type_df], axis=1)
        numeric_features += list(loan_type_df.columns)

    return X, y, numeric_features, categorical_features


def prepare_predict_data(data):
    """
    Process prediction data:
    - Drop Credit_Score column if it exists
    - Process the 'Type_of_Loan' column if present
    - Drop unnecessary columns

    Returns the processed DataFrame.
    """
    drop_columns = DROP_COLUMNS.copy()

    if "Credit_Score" in data.columns:
        data = data.drop(columns=["Credit_Score"], errors="ignore")

    loan_type_df = process_type_of_loan(data)

    data.drop(columns=drop_columns, errors="ignore", inplace=True)

    if not loan_type_df.empty:
        data = pd.concat([data, loan_type_df], axis=1)

    return data
