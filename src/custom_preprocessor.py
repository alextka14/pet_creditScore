import pandas as pd


class CustomPreprocessor:
    """
    Custom preprocessor:
    - Removes unnecessary columns (default: ID, Customer_ID, SSN, Name)
    - Fills missing numeric features with median
    - Encodes categorical features using factorize
    """
    def __init__(self, drop_cols=None):
        if drop_cols is None:
            drop_cols = ["ID", "Customer_ID", "SSN", "Name"]
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        X_ = X.copy()
        X_.drop(columns=self.drop_cols, errors="ignore", inplace=True)
        self.numeric_cols_ = X_.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols_ = X_.select_dtypes(include=["object"]).columns.tolist()
        self.median_values_ = X_[self.numeric_cols_].median()
        self.feature_names_ = X_.columns.tolist()
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_.drop(columns=self.drop_cols, errors="ignore", inplace=True)
        for col in self.numeric_cols_:
            X_[col] = X_[col].fillna(self.median_values_.get(col, 0))
        for col in self.categorical_cols_:
            X_[col] = pd.factorize(X_[col])[0]
        return X_
