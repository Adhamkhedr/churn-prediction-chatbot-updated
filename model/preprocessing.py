import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# ----------------------------------------------
# COLUMN GROUP DEFINITIONS
# ----------------------------------------------
BINARY_COLS = ['Is_Married', 'Dependents', 'Phone_Service', 'Paperless_Billing']

ONEHOT_COLS = ['Internet_Service', 'Contract', 'Payment_Method']

LABEL_COLS = ['gender', 'Dual', 'Online_Security', 'Online_Backup',
              'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies']

NUMERIC_COLS = ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges',
                'charges_per_tenure']


# ----------------------------------------------
# FEATURE ENGINEER (sklearn transformer)
# ----------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds charges_per_tenure feature. Must run before ColumnTransformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['charges_per_tenure'] = np.where(
            X['tenure'] == 0,
            X['Monthly_Charges'],
            X['Total_Charges'] / X['tenure']
        )
        return X


# ----------------------------------------------
# DATA LOADING & CLEANING
# ----------------------------------------------
def load_and_clean_data():
    """Load raw CSV and return clean X, y ready for training.

    Steps:
        1. Load raw CSV
        2. Strip column names
        3. Drop customerID
        4. Convert Total_Charges to numeric
        5. Fill blank Total_Charges with 0 (tenure=0 customers not yet billed)
        6. Separate target (Churn) from features

    Returns:
        X (DataFrame): Feature matrix (20 columns)
        y (Series): Binary target (0=No, 1=Yes)
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop customerID - unique per row, not a feature
    df.drop('customerID', axis=1, inplace=True)

    # Convert Total_Charges to numeric (11 rows have blank strings)
    df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')

    # Fill NaN Total_Charges with 0 instead of dropping
    # EDA: all 11 are tenure=0 customers not yet billed, so 0 is correct
    filled_count = df['Total_Charges'].isna().sum()
    df['Total_Charges'] = df['Total_Charges'].fillna(0)
    print(f"Filled {filled_count} blank Total_Charges with 0")

    # Separate target
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)

    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Churn rate: {y.mean():.2%}")

    return X, y


# ----------------------------------------------
# PREPROCESSOR (ColumnTransformer)
# ----------------------------------------------
def build_preprocessor(numeric_cols=None):
    """Build and return the ColumnTransformer for encoding.

    Args:
        numeric_cols: Override numeric columns (for version experiments).
                      Defaults to NUMERIC_COLS if not provided.

    Encoding strategy:
        - Binary cols (Yes/No): OrdinalEncoder -> 0/1
        - OneHot cols (multi-category): OneHotEncoder, drop first
        - Label cols (multi-category for trees): OrdinalEncoder
        - Numeric cols: passthrough (no scaling needed for tree models)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS

    preprocessor = ColumnTransformer([
        ('binary', OrdinalEncoder(
            categories=[['No', 'Yes']] * len(BINARY_COLS),
            handle_unknown='use_encoded_value', unknown_value=-1
        ), BINARY_COLS),
        ('onehot', OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        ), ONEHOT_COLS),
        ('label', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ), LABEL_COLS),
        ('numeric', 'passthrough', numeric_cols),
    ])
    return preprocessor


# ----------------------------------------------
# STANDALONE TEST
# ----------------------------------------------
if __name__ == '__main__':
    X, y = load_and_clean_data()
    print(f"\nClass distribution:\n{y.value_counts().to_string()}")
    print(f"\nColumn types:\n{X.dtypes.to_string()}")
    print(f"\nSample row:\n{X.iloc[0].to_string()}")
