import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# ----------------------------------------------
# 1) COLUMN GROUP DEFINITIONS
# ----------------------------------------------
# Groups columns into 4 lists based on how they'll be encoded (binary, onehot, label, numeric)
# 3 weak features (gender, Phone_Service, Dual) are excluded — confirmed by SHAP after training

BINARY_COLS = ['Is_Married', 'Dependents', 'Paperless_Billing']
# columns that only have Yes/No values
# Phone_Service removed — weak (EDA spread 1.78pp, perm importance 0.0077, ablation confirmed)

ONEHOT_COLS = ['Internet_Service', 'Contract', 'Payment_Method']
# columns with 3+ unordered text values (3 columns)

LABEL_COLS = ['Online_Security', 'Online_Backup',
              'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies']
# gender removed — weak (EDA spread 0.76pp, perm importance 0.0117, ablation confirmed)
# Dual removed   — weak (EDA spread 3.68pp, perm importance 0.0097, ablation confirmed)

NUMERIC_COLS = ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges',
                'charges_per_tenure']
#the final model in train_model.py overrides this with NUMERIC_COLS = ['Senior_Citizen', 'tenure', 'Monthly_Charges']



# ----------------------------------------------
# FEATURE ENGINEER (sklearn transformer) — EXPERIMENT, LATER DROPPED
# ----------------------------------------------
# WHAT WE TRIED TO DO:
# Total_Charges on its own is misleading — a customer with 24 months will naturally
# have higher total charges than one with 2 months, even if they pay the same rate.
# So we created a new feature: charges_per_tenure = Total_Charges / tenure
# This normalizes it to "average charge per month", which is a fairer comparison.
#
# HOW IT WORKS:
# np.where checks every row:
#   - If tenure == 0 (new customer, never billed) → use Monthly_Charges to avoid divide-by-zero
#   - Otherwise → Total_Charges / tenure
#
# WHY IT WAS DROPPED:
# In train_model.py we tested 3 versions of NUMERIC_COLS to find the best combination:
#   Version A: ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges']
#   Version B: ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges', 'charges_per_tenure']
#   Version C: ['Senior_Citizen', 'tenure', 'Monthly_Charges']   ← WINNER
#
# Version C won because tenure and Total_Charges are 83% correlated — meaning if you
# know how long someone has been a customer, you already almost know their total charges.
# Keeping both gives the model the same signal twice, which adds noise, not value.
# And charges_per_tenure = Total_Charges / tenure, so it's redundant for the same reason.
# tenure + Monthly_Charges alone gives the model everything it needs.
#
# NOTE: This class is kept here for reference but is NOT used in the final model.
# The final NUMERIC_COLS in train_model.py is: ['Senior_Citizen', 'tenure', 'Monthly_Charges']
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """EXPERIMENT — Adds charges_per_tenure feature. Not used in final model (dropped in Version C)."""

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
# 2) DATA LOADING & CLEANING
# ----------------------------------------------
def load_and_clean_data():
    """Load raw CSV and return clean X, y ready for training.
    Steps:
        1. Load raw CSV
        2. Strip column names
        3. Drop customerID
        4. Convert Total_Charges to numeric
        5. Fill blank Total_Charges with 0 (tenure=0 customers not yet billed)
        6. Drop weak features: Total_Charges, gender, Phone_Service, Dual
        7. Separate target (Churn) from features

    Returns:
        X (DataFrame): Feature matrix (15 columns)
        y (Series): Binary target (0=No, 1=Yes)
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Builds the file path to the CSV dynamically using the location of THIS file.
    # os.path.dirname(__file__)  → folder where preprocessing.py lives  → model/
    # '..'                       → goes one folder up                   → E& usecase/
    # 'data'                     → goes into the data folder            → E& usecase/data/
    # 'WA_Fn-...-Churn.csv'      → the actual CSV file
    #
    # WHY NOT hardcode the path like "C:\Users\adham\Desktop\E& usecase\data\..."?
    # Because that only works on my computer. If a friend clones this project,
    # their path would be different (e.g. C:\Users\Ahmed\Documents\projects\E& usecase\)
    # and the code would crash with FileNotFoundError.
    # Using __file__ makes the path relative to wherever the project actually is,
    # so it works on any machine without changing any code.

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # We found this in EDA Part 1. 
    # When we ran the dtype check and unique value count, 
    # customerID had 7,043 unique values — one per row. 
    # A column that is different for every single customer 
    # can't teach the model anything. 
    # There's no pattern to learn from it.
    #  It's just a label like a name tag, not a feature.
    df.drop('customerID', axis=1, inplace=True)

    # Convert Total_Charges to numeric (11 rows have blank strings)
    df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')

    # Fill NaN Total_Charges with 0 instead of dropping
    # EDA: all 11 are tenure=0 customers not yet billed, so 0 is correct
    # WHY we still handle this even though XGBoost supports missing values natively:
    # The blank string problem hits the pipeline BEFORE XGBoost ever sees the data.
    #
    #   Raw CSV -> pandas -> ColumnTransformer (encoders) -> XGBoost
    #
    # XGBoost handles NaN fine. The problem is "" never becomes NaN unless we
    # explicitly convert it. That conversion is what this step does.
    #
    # +---------------------------------+-------------------------------+------------------------+
    # |             Issue               |  Where it crashes/corrupts    | Has nothing to do with |
    # +---------------------------------+-------------------------------+------------------------+
    # | Blank string in numeric col     | ColumnTransformer passthrough | XGBoost                |
    # | Blank string in categorical col | OrdinalEncoder (silent -1)    | XGBoost                |
    # +---------------------------------+-------------------------------+------------------------+
    filled_count = df['Total_Charges'].isna().sum()
    df['Total_Charges'] = df['Total_Charges'].fillna(0)
    print(f"Filled {filled_count} blank Total_Charges with 0")

    # Drop confirmed weak features — evidence from 3 methods:
    #
    # Total_Charges:
    #   - 0.83 correlated with tenure (EDA Part 4C) — nearly the same signal
    #   - Permutation importance = 0.0000 — shuffling it causes zero Recall drop
    #
    # gender, Phone_Service, Dual:
    #   - EDA Part 5: churn rate spread of 0.76pp, 1.78pp, 3.68pp — near zero separation
    #   - Permutation importance: 0.0117, 0.0077, 0.0097 — small individual contribution
    #   - Ablation test: dropping all 3 together improves Recall from 0.8075 to 0.8209
    #     (individually they add slight noise; together their removal actually helps)
    df.drop(columns=['Total_Charges', 'gender', 'Phone_Service', 'Dual'], inplace=True)

    # Separate target
    y = df['Churn'].map({'Yes': 1, 'No': 0})   #Takes the Churn column and converts 'Yes' → 1 and 'No' → 0.
    X = df.drop('Churn', axis=1)

    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Churn rate: {y.mean():.2%}")

    return X, y


# ----------------------------------------------
# 4) PREPROCESSOR (ColumnTransformer)
# ----------------------------------------------
# WHY THIS FUNCTION EXISTS:
# The model can only read numbers — not text like "Yes"/"No" or "Month-to-month".
# This function builds a ColumnTransformer: a machine that applies different
# encoding rules to different columns simultaneously, converting everything to numbers.

# NOTE: This function only BUILDS the transformer — it doesn't transform anything yet.
# The actual encoding happens in train_model.py when .fit_transform() is called.
def build_preprocessor(numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS

    # ColumnTransformer applies different preprocessing steps to different columns simultaneously
    preprocessor = ColumnTransformer([
        ('binary', OrdinalEncoder(
            categories=[['No', 'Yes']] * len(BINARY_COLS),  #for each binary column, force the order No=0, Yes=1
            handle_unknown='use_encoded_value', unknown_value=-1
        ), BINARY_COLS),    #Applies OrdinalEncoder to BINARY_COLS
        ('onehot', OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        ), ONEHOT_COLS),
        ('label', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ), LABEL_COLS),
        ('numeric', 'passthrough', numeric_cols),
    ])
    return preprocessor
#returns the ColumnTransformer object but doesn't run it yet. 
#The actual encoding happens in train_model.py when .fit() is called.

# ----------------------------------------------
# STANDALONE TEST
# ----------------------------------------------
if __name__ == '__main__':
    X, y = load_and_clean_data()
    print(f"\nClass distribution:\n{y.value_counts().to_string()}")
    print(f"\nColumn types:\n{X.dtypes.to_string()}")
    print(f"\nSample row:\n{X.iloc[0].to_string()}")
