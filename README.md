# CreditScore Project

A simple credit scoring project using XGBoost and data preprocessing with Python.

## Project Structure

- **custom_preprocessor.py**  
  Contains a simple custom preprocessor class for dropping columns, filling missing numeric data, and factorizing categorical columns.

- **data_preprocessing.py**  
  Shared data processing logic (dropping unnecessary columns, processing `Type_of_Loan`, mapping `Credit_Score` to numeric values, etc.).

- **train.py**  
  Trains the XGBoost model on the credit score dataset.  
  - Uses a `ColumnTransformer` for preprocessing (scaling numeric features, one-hot encoding categorical features).  
  - Uses `SMOTE` for oversampling.  
  - Performs a `GridSearchCV` for hyperparameter tuning.  
  - Saves the final model and the preprocessor.

- **predict.py**  
  Loads the saved preprocessor and XGBoost model, then makes predictions on new data.  
  - Optionally displays evaluation metrics if `Credit_Score` is present in the input data.

## Installation

1. Clone the repository.
2. Install required packages:
   ```bash
   pip install -r requirements.txt

## Usage

1. **Train the Model**  
   ```bash
   python train.py
   ```
   This will:
   - Read data from `data/credit_score_data.csv`
   - Train the model
   - Save the preprocessor (`preprocessing_pipeline.pkl`) and the XGBoost model (`credit_rating_xgb.json`) in the `models` directory.

2. **Predict with New Data**  
   ```bash
   python predict.py path_to_new_data.csv
   ```
   This will:
   - Load the preprocessor and model from the `models` directory
   - Print predictions and, if available, compare them with the actual `Credit_Score` labels.

## Examples
The example of a successful execution is posted at the repository - **results.pdf** 