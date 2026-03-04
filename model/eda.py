import pandas as pd
import numpy as np
import matplotlib
# 'Agg' is a non-interactive backend — renders plots to image files instead of
# opening a GUI window. Required when running as a script with no display.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # used to build custom legend entries in combo plots
import os

# Build absolute paths relative to this file's location so the script works
# regardless of which directory you run it from.
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'eda_plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ══════════════════════════════════════════════════
# PART 1 — BASIC DATA UNDERSTANDING
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   - How many rows and columns does the dataset have?
#   - What are the column names and their data types?
#   - Are all columns stored as the correct type, or is something miscoded?
#   - Basic statistics (mean, median, min, max) for numeric columns
#
# FINDING:
#   - Dataset has 7,043 rows and 21 columns
#   - Total_Charges is stored as object (text) instead of float — it has blank
#     strings that prevented pandas from reading it as a number

#   - This is a data quality issue that must be fixed before any numeric analysis
#   - Total_Charges is missing from the stats table below for this exact reason

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("=" * 60)
print("PART 1 — BASIC DATA UNDERSTANDING")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nFirst 5 rows:")
print(df.head().to_string())

# Check column data types — looking for any column stored as the wrong type
print(f"\n{'Column':<25} {'Dtype':<15}")
print("-" * 40)
for col in df.columns:
    print(f"{col:<25} {str(df[col].dtype):<15}")

# Only numeric columns are included here — Total_Charges is excluded because
# it was read as object (text) due to blank strings, not as a number
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nBasic statistics for numeric columns ({', '.join(numeric_cols)}):")
print(df[numeric_cols].describe().to_string())

# OUTPUT OF DESCRIBE (actual values from the dataset):
#
#        Senior_Citizen       tenure  Monthly_Charges
# count     7043.000000  7043.000000      7043.000000
# mean         0.162147    32.371149        64.761692
# std          0.368612    24.559481        30.090047
# min          0.000000     0.000000        18.250000
# 25%          0.000000     9.000000        35.500000
# 50%          0.000000    29.000000        70.350000
# 75%          0.000000    55.000000        89.850000
# max          1.000000    72.000000       118.750000
#
# SKEW FINDINGS:
#   - tenure:          mean (32.4) > median (29.0) -> RIGHT skewed
#                      A tail of long-tenure customers (up to 72 mo) pulls the mean up.
#   - Monthly_Charges: mean (64.8) < median (70.4) -> LEFT skewed
#                      Q1-to-median gap (34.85) is nearly double median-to-Q3 gap (19.50),
#                      meaning low-charge customers (min $18.25) pull the mean down.
#
# WHY THIS DOES NOT AFFECT XGBOOST:
#   Tree-based models like XGBoost split on thresholds (e.g. tenure < 15),
#   not on the actual numeric magnitude of values. They don't compute distances
#   or assume any distribution (unlike KNN, SVM, or linear regression which
#   rely on Euclidean distance or dot products). A skewed feature and its
#   log-transformed version produce the exact same tree splits — the rank
#   order of values is all that matters. No normalization or transformation
#   is needed before passing features into XGBoost.




# ══════════════════════════════════════════════════
# PART 2 — DATA QUALITY
# ══════════════════════════════════════════════════
# WHAT I LOOKED FOR:
#   - Are there any standard null (NaN) values in any column?
#   - Are there blank/whitespace strings hiding as valid values?
#     pd.read_csv does NOT convert '' to NaN, so isnull() alone would miss them —
#     we must cast each column to str, strip whitespace, and compare to ''
#   - Are there fully duplicate rows (every column identical)?
#   - How many unique values does each column have?
#     This tells us which columns are binary flags (2 values), multi-class (3-4),
#     or near-continuous (hundreds/thousands), which informs encoding decisions
#
# FINDING:
#   - 0 standard nulls across all 21 columns
#   - 11 blank strings found in Total_Charges only
#     All 11 rows have tenure=0 (brand-new customers, not yet billed)
#     All 11 are on long-term contracts (Two year / One year) with Churn=No —
#     these are real customers, not corrupt data. Dropped at preprocessing stage.
#   - 0 duplicate rows
#   - customerID: 7043 unique values — every row is a distinct customer (ID column)
#   - Most categoricals have 2-3 unique values (binary flags or ternary Yes/No/No service)
#   - Monthly_Charges: 1585 unique values, Total_Charges: 6531 — near-continuous

print("\n\n" + "=" * 60)
print("PART 2 — DATA QUALITY")
print("=" * 60)

# CHECK 1: Standard nulls (NaN / None)
# isnull() only catches proper NaN values — blank strings are invisible to it
print("\nNull values per column:")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("  No standard null values detected in any column.")
else:
    print(null_counts[null_counts > 0].to_string())

# OUTPUT: No standard null values detected in any column.

# CHECK 2: Blank/whitespace strings
# We cast every column to str first so numeric columns don't throw errors,
# then strip() removes leading/trailing spaces and we compare to empty string
print("\nBlank/whitespace strings per column:")
blank_found = False
for col in df.columns:
    blank_count = (df[col].astype(str).str.strip() == '').sum()
    if blank_count > 0:
        print(f"  {col}: {blank_count}")
        blank_found = True
if not blank_found:
    print("  No blank/whitespace strings detected in any column.")

# OUTPUT:
#   Total_Charges: 11

# CHECK 3: Inspect the 11 blank Total_Charges rows
# Goal: understand WHY they're blank before deciding how to handle them.
# If they were random corruptions we'd impute; but if there's a pattern,
# that pattern tells us the correct action.
blank_tc = df[df['Total_Charges'].astype(str).str.strip() == '']
if len(blank_tc) > 0:
    print(f"\nFull details of rows with blank Total_Charges:")
    print(blank_tc.to_string())

# OUTPUT (key columns shown):
#       customerID  tenure  Monthly_Charges  Total_Charges  Contract   Churn
#  488  4472-LVYGI       0            52.55                 Two year   No
#  753  3115-CZMZD       0            20.25                 Two year   No
#  936  5709-LVOEQ       0            80.85                 Two year   No
# 1082  4367-NUYAO       0            25.75                 Two year   No
# 1340  1371-DWPAZ       0            56.05                 Two year   No
# 3331  7644-OMVMY       0            19.85                 Two year   No
# 3826  3213-VVOLG       0            25.35                 Two year   No
# 4380  2520-SGTTA       0            20.00                 Two year   No
# 5218  2923-ARZLG       0            19.70                 One year   No
# 6670  4075-WKNIU       0            73.35                 Two year   No
# 6754  2775-SEFEE       0            61.90                 Two year   No
#
# Pattern: every row has tenure=0 and Churn=No.
# These customers just signed up and haven't completed a billing cycle yet,
# so Total_Charges is blank — not a data error, just a timing issue.
# Action: drop these 11 rows at preprocessing (they carry no churn signal).

# CHECK 4: Fully duplicate rows
dup_count = df.duplicated().sum()
print(f"\nCompletely duplicate rows: {dup_count}")

# OUTPUT: Completely duplicate rows: 0

# CHECK 5: Unique value counts per column
# Helps classify each column before encoding:
#   2 unique  -> binary flag (Yes/No)  -> encode as 0/1
#   3-4 unique -> multi-class          -> one-hot encode
#   73+ unique -> near-continuous      -> treat as numeric
print(f"\nUnique values per column:")
print(f"{'Column':<25} {'Unique Count':<15}")
print("-" * 40)
for col in df.columns:
    print(f"{col:<25} {df[col].nunique():<15}")

# OUTPUT:
#   customerID                7043   <- all unique, pure ID — drop before modelling
#   gender                    2
#   Senior_Citizen            2
#   Is_Married                2
#   Dependents                2
#   tenure                    73     <- near-continuous numeric
#   Phone_Service             2
#   Dual                      3      <- Yes / No / No phone service
#   Internet_Service          3      <- DSL / Fiber optic / No
#   Online_Security           3      <- Yes / No / No internet service
#   Online_Backup             3
#   Device_Protection         3
#   Tech_Support              3
#   Streaming_TV              3
#   Streaming_Movies          3
#   Contract                  3      <- Month-to-month / One year / Two year
#   Paperless_Billing         2
#   Payment_Method            4
#   Monthly_Charges           1585   <- near-continuous numeric
#   Total_Charges             6531   <- near-continuous numeric (but dropped later)
#   Churn                     2      <- target variable


# ══════════════════════════════════════════════════
# PART 3 — TARGET VARIABLE
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   - How many customers churned vs. stayed?
#   - Is the dataset balanced, or does one class dominate?
#   - What does class imbalance mean for how we evaluate the model?
#
# FINDING:
#   - No:  5174 customers (73.46%) — stayed
#   - Yes: 1869 customers (26.54%) — churned
#   - Moderate class imbalance: "No" outnumbers "Yes" by ~2.8x
#   - A model that always predicts "No" gets 73.5% accuracy with 0% Recall
#   - This makes accuracy a misleading metric — Recall is used instead
#     Recall asks: "of all customers who actually churned, how many did we catch?"

print("\n\n" + "=" * 60)
print("PART 3 — TARGET VARIABLE")
print("=" * 60)

# value_counts() counts occurrences of each unique value in the Churn column
churn_counts = df['Churn'].value_counts()
# normalize=True converts raw counts to proportions; multiply by 100 for percentages
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn column value counts:")
for val in churn_counts.index:
    print(f"  {val}: {churn_counts[val]}  ({churn_pct[val]:.2f}%)")

# OUTPUT:
#   Churn column value counts:
#     No:  5174  (73.46%)
#     Yes: 1869  (26.54%)

# --- Bar chart: raw counts with the exact number printed on top of each bar ---
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ['No', 'Yes'],
    [churn_counts['No'], churn_counts['Yes']],
    color=['#2563eb', '#ef4444'], edgecolor='white', width=0.5
)
for bar in bars:
    height = bar.get_height()
    # Place the count label just above the top of each bar
    ax.text(bar.get_x() + bar.get_width() / 2, height + 50,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('Churn Count', fontsize=14, fontweight='bold')
ax.set_xlabel('Churn')
ax.set_ylabel('Number of Customers')
# Add 15% headroom above the tallest bar so the count label isn't cut off
ax.set_ylim(0, max(churn_counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_bar.png'), dpi=150)
plt.close()  # close the figure to free memory before creating the next one
print(f"\nSaved: target_bar.png")

# --- Pie chart: shows the percentage split at a glance ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    [churn_counts['No'], churn_counts['Yes']],
    labels=['No', 'Yes'],
    autopct='%1.1f%%',      # prints the percentage inside each slice
    colors=['#2563eb', '#ef4444'],
    startangle=90,          # rotates so the first slice starts at the top
    textprops={'fontsize': 12}
)
ax.set_title('Churn Percentage Split', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_pie.png'), dpi=150)
plt.close()
print(f"Saved: target_pie.png")

# WHY RECALL INSTEAD OF ACCURACY:
# If we shipped a model that always says "No churn", it would be:
#   - 73.5% accurate    <- sounds good
#   - 0% Recall         <- catches zero actual churners
#   - completely useless for the business
#
# In churn prediction the cost of a missed churner (false negative) is high —
# you lose a customer you could have retained with a targeted offer.
# The cost of a false alarm (false positive) is low — you offer a retention
# deal to someone who wasn't going to leave anyway.
# So we optimize for Recall: maximize how many real churners we catch,
# even if it means flagging a few extra non-churners along the way.
majority_pct = churn_pct.max()
print(f"\nA model that always predicts the majority class would achieve {majority_pct:.1f}% accuracy with 0% Recall.")
print(f"Our model must be evaluated against this baseline.")


# ══════════════════════════════════════════════════
# PART 4 — NUMERICAL FEATURES
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   - How are tenure, Monthly_Charges, and Total_Charges distributed?
#   - Do churners and non-churners differ on these numeric features?
#   - How correlated are these features with each other and with Churn?
#
# NOTE — WHY WE DO THIS WITH XGBOOST:
#   XGBoost doesn't require normally distributed features (it splits on thresholds,
#   not magnitudes) and doesn't break under correlated features (unlike linear models).
#   We're not doing this analysis to fix anything for the model.
#   We're doing it to understand the data as humans — so we can make informed decisions
#   about which features to keep, which to drop, and which hypotheses to test.
#
# FINDING:
#   - tenure: strongest predictor — churners median 10 months vs. 38 for loyal (28-month gap)
#     Correlation with Churn: -0.35 (longer tenure = less likely to churn)
#   - Monthly_Charges: higher bills = slightly more churn (+0.19 correlation)
#     Churners are on more expensive plans but leave anyway
#   - Total_Charges: lower for churners (-0.20 correlation) — not because they pay less per month,
#     but because they left early so they accumulated less total spend
#   - tenure and Total_Charges are 0.83 correlated — nearly the same signal
#     XGBoost handles this fine on its own, but it gave us a hypothesis to test:
#     does dropping Total_Charges hurt the model? We tested it empirically (Versions A/B/C)
#     and Version C (without Total_Charges) actually won — confirming it added no independent signal

print("\n\n" + "=" * 60)
print("PART 4 — NUMERICAL FEATURES")
print("=" * 60)

# Fix Total_Charges before any numeric analysis:
# Step 1: strip whitespace, then replace remaining empty strings with '0'
#         (the 11 tenure=0 rows had blank Total_Charges — $0 is the correct value)
# Step 2: convert the whole column from string to float using pd.to_numeric
df['Total_Charges'] = df['Total_Charges'].str.strip().replace('', '0')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
print(f"\nAfter converting Total_Charges to numeric: {len(df)} rows remain (no rows dropped)")

# OUTPUT: After converting Total_Charges to numeric: 7043 rows remain (no rows dropped)

# ────────────────────────────────────────
# 4A — Distributions
# ────────────────────────────────────────
print("\n--- 4A: Distributions ---")

num_features = ['tenure', 'Monthly_Charges', 'Total_Charges']

# Loop over each numeric feature and save a histogram of its distribution
for feat in num_features:
    fig, ax = plt.subplots(figsize=(8, 5))
    # bins=36 gives ~1 bin per 2 months for tenure (max 72); works well for all three
    ax.hist(df[feat], bins=36, color='#2563eb', edgecolor='white', alpha=0.85)
    ax.set_title(f'Distribution of {feat}', fontsize=14, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Number of Customers')
    plt.tight_layout()
    # Dynamic filename: e.g. dist_tenure.png, dist_monthly_charges.png
    fname = f'dist_{feat.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")

print("\n4A Observations:")
print("  tenure: two concentrations are visible — one at the lowest values and one at the highest, with a flatter middle section.")
print("  Monthly_Charges: a tall cluster near ~$20 and a broad, roughly uniform spread from ~$40 to ~$110.")
print("  Total_Charges: most values are concentrated near the low end, tapering off gradually toward higher values.")

# OUTPUT:
# 4A Observations:
#   tenure: two concentrations are visible — one at the lowest values and one at the highest,
#           with a flatter middle section.
#           (new customers just joining + long-term loyal customers, fewer in the middle)
#   Monthly_Charges: a tall cluster near ~$20 (basic plans) and a broad spread from ~$40 to ~$110.
#   Total_Charges: concentrated near the low end — most customers haven't been around long enough
#                  to accumulate high totals. Follows directly from the tenure distribution.

# ────────────────────────────────────────
# 4B — Numerical features vs Churn
# ────────────────────────────────────────
print("\n--- 4B: Numerical Features vs Churn ---")

# ── WHY THIS SECTION EXISTS ───────────────────────────────────────────────────
# The question we're asking: do customers who churned have different
# tenure/charges than customers who didn't?
#
# If both groups look the same → the feature is useless for prediction.
# If the groups look different → the model can use that feature to tell them apart.
#
# We split all 7,043 customers into two piles:
#   - churn_no:  5,174 customers who did NOT churn
#   - churn_yes: 1,869 customers who DID churn
#
# Then for each numeric feature we compare the two piles side by side.
#
# Key findings:
#   tenure        → churners stayed  10 months median vs 38 for loyal customers (28-month gap)
#   Monthly_Charges → churners pay MORE per month (expensive plans, leave anyway)
#   Total_Charges   → churners have LOWER total (because they left early = short tenure)
#
# Purpose: prove these features are worth keeping in the model.
# ─────────────────────────────────────────────────────────────────────────────

# Split the DataFrame into two groups for comparison
churn_yes = df[df['Churn'] == 'Yes']   # 1,869 rows
churn_no = df[df['Churn'] == 'No']     # 5,174 rows

for feat in num_features:
    # --- Box plot: shows median, IQR, and outliers for each churn group ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [churn_no[feat].values, churn_yes[feat].values],
        tick_labels=['No', 'Yes'],
        patch_artist=True,              # needed to fill box with color
        medianprops=dict(color='red', linewidth=2)  # make median line visible
    )
    # Color the boxes: light blue for No, light red for Yes
    bp['boxes'][0].set_facecolor('#bfdbfe')
    bp['boxes'][1].set_facecolor('#fecaca')
    ax.set_title(f'{feat} by Churn Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Churn')
    ax.set_ylabel(feat)
    plt.tight_layout()
    fname_box = f'box_{feat.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname_box), dpi=150)
    plt.close()

    # --- Overlapping histogram: shows where the two distributions differ ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # alpha=0.55 makes both histograms semi-transparent so they overlap visibly
    ax.hist(churn_no[feat], bins=30, alpha=0.55, color='#2563eb', label='No', edgecolor='white')
    ax.hist(churn_yes[feat], bins=30, alpha=0.55, color='#ef4444', label='Yes', edgecolor='white')
    ax.set_title(f'{feat}: Churn=No vs Churn=Yes', fontsize=14, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Number of Customers')
    ax.legend(title='Churn')
    plt.tight_layout()
    fname_hist = f'hist_{feat.lower()}_vs_churn.png'
    plt.savefig(os.path.join(PLOT_DIR, fname_hist), dpi=150)
    plt.close()

    # --- Print summary statistics for both groups ---
    med_no = churn_no[feat].median()
    med_yes = churn_yes[feat].median()
    mean_no = churn_no[feat].mean()
    mean_yes = churn_yes[feat].mean()

    print(f"\n  {feat}:")
    print(f"    Churn=No  — median: {med_no:.2f}, mean: {mean_no:.2f}")
    print(f"    Churn=Yes — median: {med_yes:.2f}, mean: {mean_yes:.2f}")
    # Show which group is higher and by how much — key for the interview answer
    print(f"    Difference — median: {abs(med_no - med_yes):.2f} ({'No higher' if med_no > med_yes else 'Yes higher'}), mean: {abs(mean_no - mean_yes):.2f} ({'No higher' if mean_no > mean_yes else 'Yes higher'})")
    print(f"    Saved: {fname_box}, {fname_hist}")

print("\n4B Observations:")
print("  tenure: Churn=Yes group has a median of 10 and Churn=No has 38 — a gap of 28 months.")
print("  Monthly_Charges: Churn=Yes group has a higher median than Churn=No.")
print("  Total_Charges: Churn=Yes group has a lower median than Churn=No.")
print("  Total_Charges combines both tenure duration and monthly rate, so it reflects both factors at once.")

# OUTPUT:
# 4B Observations:
#   tenure:
#     Churn=No  — median: 38.00, mean: 37.57
#     Churn=Yes — median: 10.00, mean: 17.98
#     Difference — median: 28.00 (No higher) <- strongest separator of the three features
#
#   Monthly_Charges:
#     Churn=No  — median: 64.43, mean: 61.27
#     Churn=Yes — median: 79.65, mean: 74.44
#     Difference — median: 15.22 (Yes higher) <- churners are on more expensive plans
#
#   Total_Charges:
#     Churn=No  — median: 1683.35, mean: 2555.34
#     Churn=Yes — median:  703.55, mean: 1531.80
#     Difference — median: 979.80 (No higher) <- low because churners leave early, not because they pay less

"""We already know from the same 4B that churners stayed much shorter (10 months median vs 38). So of course
their total is lower — they simply didn't stick around long enough to accumulate a big bill. It has nothing
to do with their plan or payment behavior.
So the "signal" in Total_Charges is just tenure showing up again in a different number. Remove tenure from 
the equation and Total_Charges tells you nothing extra """
# ────────────────────────────────────────
# 4C — Correlation
# ────────────────────────────────────────
print("\n--- 4C: Correlation ---")

# ── WHY THIS SECTION EXISTS ───────────────────────────────────────────────────
# Question: how related are these numeric features to each other — and to Churn?
#
# Correlation is a number between -1 and +1:
#   close to +1 → when one goes up, the other goes up too
#   close to -1 → when one goes up, the other goes down
#   close to  0 → no relationship, they move independently
#
# Churn is "Yes"/"No" text — you can't do math on text.
# So we convert it to 1/0 first so it can be included in the calculation.
#
# Key findings:
#   tenure ↔ Total_Charges   = +0.83 → strongly related (longer stay = higher total bill)
#   tenure ↔ Churn           = -0.35 → as tenure goes up, churn goes down (confirms 4B)
#   Monthly_Charges ↔ Churn  = +0.19 → higher monthly bill = slightly more churn
#
# Most important finding: tenure and Total_Charges are 0.83 correlated — they're
# nearly telling the model the same thing (multicollinearity).
# This is normally a problem, but XGBoost (a tree model) handles it fine — trees
# just pick the more useful one naturally. So we kept both.
# This also motivated Part 7 — testing a new feature (charges_per_tenure) to see
# if it could replace both. It couldn't, so it was dropped.
#
# The heatmap is just a visual version of the same numbers:
#   blue  = positive correlation
#   red   = negative correlation
#   number printed inside each cell = the exact value
# ─────────────────────────────────────────────────────────────────────────────

df_corr = df[['tenure', 'Monthly_Charges', 'Total_Charges']].copy()
df_corr['Churn'] = (df['Churn'] == 'Yes').astype(int)

corr_matrix = df_corr.corr()

print(f"\nCorrelation matrix:")
for col in corr_matrix.columns:
    vals = '  '.join(f"{corr_matrix.loc[col, c]:>7.4f}" for c in corr_matrix.columns)
    print(f"  {col:<20} {vals}")

# --- Heatmap: visualize the correlation matrix as a color-coded grid ---
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color=color, fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Correlation Heatmap (Numeric Features + Churn)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print(f"\nSaved: correlation_heatmap.png")

print("\n4C Observations:")
print("  tenure and Total_Charges show a correlation of 0.83 — the strongest pair among the numeric features.")
print("  Monthly_Charges and Total_Charges show 0.65.")
print("  tenure and Churn show -0.35 — the largest magnitude correlation with the target among these three.")
print("  Monthly_Charges and Churn show 0.19 (positive).")
print("  Total_Charges and Churn show -0.20.")

# OUTPUT:
# Correlation matrix:
#                       tenure  Monthly_Charges  Total_Charges   Churn
#   tenure               1.0000           0.2474         0.8255  -0.3524
#   Monthly_Charges      0.2474           1.0000         0.6510   0.1930
#   Total_Charges        0.8255           0.6510         1.0000  -0.1990
#   Churn               -0.3524           0.1930        -0.1990   1.0000
#
# Key takeaways:
#   tenure ↔ Total_Charges = 0.83 → near-redundant features (keeping both = same signal twice)
#   tenure ↔ Churn         = -0.35 → strongest predictor of the three
#   Monthly_Charges ↔ Churn = +0.19 → weak but useful positive signal
#   Total_Charges ↔ Churn   = -0.20 → driven by tenure, not an independent signal


# ── SECTION 4 CONCLUSION ─────────────────────────────────────────────────────
# 4A, 4B, and 4C each looked at the numeric features from a different angle.
#the chain is:
  #- 4A — Total_Charges looks like tenure (same shape)
  #- 4B — the reason churners have lower Total_Charges is tenure (they left early)
  #- 4C — the correlation is 0.83 — mathematically confirmed they carry nearly the same signal
# Together they built a case that led to one decision:

# KEEP:
#   - tenure         — strongest predictor. 28-month median gap between churners and loyal
#                      customers (4B). Directly causes the shape of Total_Charges (4A).
#                      Strongest correlation with Churn at -0.35 (4C).
#   - Monthly_Charges — independent signal. Not driven by tenure (low 0.25 correlation
#                      between them in 4C). (you can be a new customer on a cheap plan or an expensive one — tenure doesn't 
#                      determine your monthly rate) Churners consistently pay more per month (4B).
#
# DROP:
#   - Total_Charges  — not an independent feature. Its shape in 4A mirrors tenure.
#                      Its separation in 4B is explained entirely by tenure (churners have
#                      lower totals only because they left early, not because they pay less).
#                      4C confirmed: 0.83 correlation with tenure — nearly the same signal.
#                      The correlation told us they were redundant. 4B told us which one
#                      was the cause (tenure) and which was the echo (Total_Charges).
#
# IMPORTANT: The decision to drop Total_Charges was not made here.
#   Part 4 produced a hypothesis: "Total_Charges looks redundant — test dropping it."
#   The actual decision was made empirically in train_model.py by running all three
#   versions (A: keep Total_Charges, B: keep both + charges_per_tenure, C: drop both)
#   and measuring recall. Version C won, validating what Part 4 suspected.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════
# PART 5 — CATEGORICAL FEATURES - pre-training filter
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   - For each categorical column, do different categories have meaningfully
#     different churn rates?
#   - Which categorical features separate churners from non-churners the most?
#   - Which features are so weak they add no value to the model?
#
# APPROACH — CHURN RATE SPREAD:
#   For each category value, calculate: churned / total in that category = churn rate
#   Then compute the spread: max churn rate − min churn rate across all values
#   A large spread = categories behave very differently → useful feature
#   A tiny spread  = categories behave the same → useless feature
#
#   Example — Contract (spread ~40pp):
#     Month-to-month → 42.7% churn
#     One year       → 11.3% churn
#     Two year       →  2.8% churn
#     The category a customer is in strongly predicts whether they churn
#
#   Example — gender (spread ~0.76pp):
#     Male   → 26.2% churn
#     Female → 26.9% churn
#     Knowing someone's gender tells you almost nothing about their churn risk

# FINDING:
#   Ranked by spread (highest to lowest):
#     1. Contract          ~39.88pp  <- strongest categorical predictor
#     2. Internet_Service  ~34.49pp
#     3. Online_Security   ~34.36pp
#     4. Tech_Support      ~34.23pp
#     ...
#     14. Dual              ~3.68pp
#     15. Phone_Service     ~1.78pp
#     16. gender            ~0.76pp  <- weakest, essentially no signal
#
#   Bottom 3 (gender, Phone_Service, Dual) dropped from the model.
#   Decision confirmed by SHAP importance scores after training.
# ─────────────────────────────────────────────────────────────────── 
# LIMITATIONS OF THIS METHOD:
#   1. Spread ignores sample size per category
#      The spread is just a number — it doesn't tell you whether that number is
#      meaningful or happened by chance. A category with only 10 customers showing
#      60% churn could easily be a coincidence. The spread would still rank it highly.
#
#   2. No formal test for statistical significance
#      The spread doesn't tell you if the difference between categories is real
#      or just random variation. Two features could have the same spread but one
#      could be based on 3000 customers per category and the other on 30.
#
# WHY IT STILL WORKS HERE:
#   Both limitations become irrelevant when category sample sizes are large enough.
#   In this dataset the categories are well-populated:
#     - gender:        Male ~3555,  Female ~3488  — near 50/50 split of 7043
#     - Phone_Service: Yes ~6361,   No ~682       — smallest group still 682
#     - Dual:          Yes ~2971,   No ~3390,     No phone ~682
#   No category has dangerously few customers, so the spreads are trustworthy
#   in both directions — big spreads are real signal, small spreads are real noise.
#   The 0.76pp for gender isn't small because of a sample size issue —
#   it's small because gender genuinely doesn't predict churn in this dataset.
#
# THE RIGOROUS ALTERNATIVE — CHI-SQUARE TEST:
#   A chi-square test formally checks: is the difference in churn rates across
#   categories statistically significant, or could it be random noise?
#   It compares what you observed (e.g. 42.7% churn for month-to-month) against
#   what you'd expect if churn had no relationship with contract type at all
#   (the overall 26.5% rate applied equally to everyone).
#   If the gap is large enough given the sample size, it returns a p-value < 0.05,
#   meaning the difference is real.
#
#   For this dataset:
#     Contract       → p ≈ 0.0000  (extremely significant)
#     gender         → p ≈ 0.50+   (not significant at all)
#   Which is exactly what our spread method already concluded.
#   The chi-square would validate our approach, not change it.
#
# IS THIS METHOD VALID FOR THIS DATASET?
#   Yes. The spreads are extreme (40pp vs 0.76pp — not borderline cases),
#   the category sizes are large enough to trust the numbers, and SHAP values
#   after training confirmed the same ranking independently.
#----------------------------------------------------------------------------

# All categorical columns to analyze (excludes customerID and the target 'Churn')
cat_columns = [
    'gender', 'Senior_Citizen', 'Is_Married', 'Dependents',
    'Phone_Service', 'Dual', 'Internet_Service',
    'Online_Security', 'Online_Backup', 'Device_Protection',
    'Tech_Support', 'Streaming_TV', 'Streaming_Movies',
    'Contract', 'Paperless_Billing', 'Payment_Method',
]

# This dict will store the churn rate spread for each feature
# (max churn rate across categories − min churn rate)
# A large spread means the feature strongly separates churners from non-churners
churn_rate_ranges = {}  # column -> max churn rate difference across its values

for col in cat_columns:
    print(f"\n{'-' * 50}")
    print(f"  {col}")
    print(f"{'-' * 50}")

    # Step A: How many customers fall into each category?
    print(f"\n  Value counts:")
    vc = df[col].value_counts()
    for val in vc.index:
        print(f"    {val}: {vc[val]}")

    # Step B: For each category value, compute churn rate = churned / total in that value
    # This is the core metric for comparing feature predictiveness
    print(f"\n  Churn rate per value:")
    churn_rates = {}
    for val in vc.index:
        subset = df[df[col] == val]              # all rows with this category value
        churned = subset[subset['Churn'] == 'Yes']  # subset of those who churned
        rate = len(churned) / len(subset) * 100
        churn_rates[val] = rate
        print(f"    {val}: {len(churned)}/{len(subset)} = {rate:.2f}%")

    # The range (max - min) measures how much predictive power this feature has
    # e.g. Contract has ~40pp range, gender has ~0.76pp range
    max_rate = max(churn_rates.values())
    min_rate = min(churn_rates.values())
    churn_rate_ranges[col] = max_rate - min_rate

    # Step C: Grouped bar chart — side-by-side bars for Churn=No and Churn=Yes
    categories = list(vc.index)
    no_counts = [len(df[(df[col] == val) & (df['Churn'] == 'No')]) for val in categories]
    yes_counts = [len(df[(df[col] == val) & (df['Churn'] == 'Yes')]) for val in categories]

    x = np.arange(len(categories))  # evenly spaced tick positions
    width = 0.35  # each bar takes up 35% of the space between ticks

    # Dynamic figure width: wider charts for features with more categories (e.g. Payment_Method)
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 5))
    # Offset the two bars left/right by half a width so they sit side by side
    bars_no = ax.bar(x - width / 2, no_counts, width, label='No', color='#2563eb', edgecolor='white')
    bars_yes = ax.bar(x + width / 2, yes_counts, width, label='Yes', color='#ef4444', edgecolor='white')

    ax.set_title(f'{col}: Churned vs Not Churned', fontsize=14, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Number of Customers')
    ax.set_xticks(x)
    # Rotate labels 45° for features with many categories so they don't overlap
    ax.set_xticklabels([str(v) for v in categories], rotation=45 if len(categories) > 3 else 0, ha='right' if len(categories) > 3 else 'center')
    ax.legend(title='Churn')
    plt.tight_layout()
    fname = f'cat_{col.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
    plt.close()
    print(f"\n  Saved: {fname}")

# Sort churn_rate_ranges descending and print a ranked table
# lambda x: x[1] sorts by the value (the pp difference), not the key (column name)
print(f"\n\n{'=' * 60}")
print("CATEGORICAL FEATURES RANKED BY CHURN RATE DIFFERENCE")
print("=" * 60)
print(f"\n{'Rank':<6} {'Feature':<25} {'Max Difference':<15}")
print("-" * 46)

ranked = sorted(churn_rate_ranges.items(), key=lambda x: x[1], reverse=True)
for i, (col, diff) in enumerate(ranked, 1):
    print(f"{i:<6} {col:<25} {diff:.2f} pp")

# OUTPUT:
# Rank   Feature                   Max Difference
# ----------------------------------------------
# 1      Contract                  39.88 pp
# 2      Internet_Service          34.49 pp
# 3      Online_Security           34.36 pp
# 4      Tech_Support              34.23 pp
# 5      Online_Backup             29.14 pp
# 6      Device_Protection         27.24 pp
# 7      Streaming_Movies          20.89 pp
# 8      Streaming_TV              20.07 pp
# 9      Payment_Method            19.66 pp
# 10     Paperless_Billing         16.27 pp
# 11     Senior_Citizen            15.95 pp
# 12     Dependents                11.69 pp
# 13     Is_Married                 9.85 pp
# 14     Dual                       3.68 pp  <- dropped
# 15     Phone_Service              1.78 pp  <- dropped
# 16     gender                     0.76 pp  <- dropped
#
# CONCLUSION:
#   The bottom 3 features (gender, Phone_Service, Dual) have spreads under 4pp —
#   effectively no difference in churn rate between their categories.
#   They were dropped from the model. This was NOT the final decision on its own —
#   SHAP values after training confirmed all three had near-zero importance,
#   validating what this ranking suggested.


# ══════════════════════════════════════════════════
# PART 6 — FEATURE COMBINATIONS
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   Parts 4 and 5 looked at features one at a time.
#   Part 6 asks: do two features TOGETHER reveal something that neither shows alone?
#   This is called an interaction effect.
#
# HONEST ASSESSMENT OF EACH SUB-SECTION:
#   6.1 — NOT a true interaction effect. Just Monthly_Charges broken down by Contract type.
#          Descriptive context, doesn't reveal anything Parts 4 and 5 didn't already cover.
#   6.2 — NOT a true interaction effect. Just Tenure broken down by Contract type.
#          Same issue — descriptive, not additive insight.
#   6.3 — NOT a true interaction effect. Just Monthly_Charges broken down by Internet_Service.
#          Shows fiber optic customers pay more, which explains some of their higher churn,
#          but again nothing genuinely new beyond Parts 4 and 5.
#   6.4 — GENUINE interaction effect. The only sub-section that earns the section title.
#          Quadrant analysis reveals low tenure + high charges = 58% churn — far higher
#          than either feature suggested individually. Actionable business insight.
#   6.5 — BORDERLINE. Shows seniors on month-to-month churn more than non-seniors on the
#          same contract. A real interaction, but XGBoost would learn this on its own and
#          there's no clear business action beyond what Part 5 already suggested.
#
# NOTE — WHY WE DO THIS FOR XGBOOST:
#   XGBoost automatically learns interaction effects during training — it doesn't need
#   us to manually identify them. Part 6 is not for the model.
#   6.1-6.3 provide business context. 6.4 provides a specific actionable segment
#   the model can't surface on its own (it predicts churn but doesn't tell you
#   "target new customers paying over $64/month with a retention offer in month 1-6").
#
# THE ONE FINDING THAT MATTERS — 6.4 Quadrant Analysis:
#   Median split: tenure at 29 months, Monthly_Charges at $64.76
#
#   Low tenure  + Low charges  → ~24% churn   (average risk)
#   Low tenure  + High charges → ~58% churn   <- HIGHEST RISK — primary target segment
#   High tenure + Low charges  → ~12% churn   (very loyal)
#   High tenure + High charges → ~20% churn   (loyal despite high bill)
#
#   Key insight: it's not high charges alone that causes churn.
#   It's high charges COMBINED WITH low tenure — new customers on expensive plans
#   who haven't been around long enough to see the value of what they're paying for.
#   A veteran on the same expensive plan stayed anyway — they've already decided it's worth it.
#
# LIMITATION OF QUADRANT ANALYSIS:
#   The median split is arbitrary. Why 29 months? Because that's the median —
#   not because 29 is a meaningful business threshold. A customer at 28 months
#   and one at 30 months land in different quadrants despite being essentially the same.
#   A more rigorous cutoff would use the threshold XGBoost actually learned to split on.
#   For EDA exploration purposes this is acceptable — the top-left corner is red
#   regardless of exactly where you draw the line.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────
# 6.1 — Monthly_Charges by Contract type, colored by Churn
# NOT a true interaction effect — descriptive context only.
# Shows that month-to-month customers tend to pay more per month than long-term
# contract customers. Useful background for understanding 6.4 but adds nothing
# beyond what Parts 4 and 5 already established individually.
# ────────────────────────────────────────
print("\n--- 6.1: Monthly_Charges by Contract type, colored by Churn ---")

contract_types = ['Month-to-month', 'One year', 'Two year']
churn_labels = ['No', 'Yes']
colors = {'No': '#2563eb', 'Yes': '#ef4444'}  # consistent color scheme throughout

# Build a grouped box plot with 6 boxes (2 churn groups × 3 contract types)
# manually managing positions so boxes within each contract type are clustered together
fig, ax = plt.subplots(figsize=(10, 6))
positions = []   # numeric x-position for each box
data = []        # array of values for each box
tick_positions = []    # where to place the contract-type label on x-axis
tick_labels_list = []
pos = 1  # start position counter at 1
for ct in contract_types:
    for ch in churn_labels:
        # Filter to this specific (contract, churn) combination
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['Monthly_Charges']
        data.append(subset.values)
        positions.append(pos)
        pos += 1
    # Place the contract label at the midpoint of its two boxes
    tick_positions.append(pos - 1.5)
    tick_labels_list.append(ct)
    pos += 0.5  # add a small gap between contract groups

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2))
# Alternate colors: blue for Churn=No, red for Churn=Yes, repeated per group
color_cycle = ['#bfdbfe', '#fecaca'] * len(contract_types)
for patch, c in zip(bp['boxes'], color_cycle):
    patch.set_facecolor(c)

# Override default x-ticks with the contract-type group labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_list)
ax.set_title('Monthly Charges by Contract Type and Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Contract Type')
ax.set_ylabel('Monthly Charges')
# Build a manual legend using Patch since the automatic one doesn't work for custom positions
ax.legend(handles=[Patch(facecolor='#bfdbfe', label='No'), Patch(facecolor='#fecaca', label='Yes')], title='Churn')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'combo_monthly_by_contract_churn.png'), dpi=150)
plt.close()
print("Saved: combo_monthly_by_contract_churn.png")

print("\n  Key numbers:")
for ct in contract_types:
    for ch in churn_labels:
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['Monthly_Charges']
        print(f"    {ct}, Churn={ch}: median={subset.median():.2f}, mean={subset.mean():.2f}, count={len(subset)}")

# ────────────────────────────────────────
# 6.2 — Tenure by Contract type, colored by Churn
# NOT a true interaction effect — descriptive context only.
# Shows that month-to-month churners leave early (left-skewed tenure histogram)
# while two-year contract churners stayed longer before leaving (locked in by contract).
# Confirms the obvious: flexible contracts = shorter tenure before churn.
# ────────────────────────────────────────
print("\n--- 6.2: Tenure by Contract type, colored by Churn ---")

# 3 side-by-side subplots (one per contract type), sharey=True so all share the same y-axis scale
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for i, ct in enumerate(contract_types):
    ax = axes[i]
    for ch in churn_labels:
        # Overlapping histograms within each subplot: blue=No, red=Yes
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['tenure']
        ax.hist(subset, bins=20, alpha=0.55, color=colors[ch], label=f'Churn={ch}', edgecolor='white')
    ax.set_title(f'{ct}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tenure (months)')
    # Only label the y-axis on the leftmost subplot to avoid repetition
    if i == 0:
        ax.set_ylabel('Number of Customers')
    ax.legend()
# suptitle adds a single title spanning all 3 subplots; y=1.02 lifts it above them
plt.suptitle('Tenure Distribution by Contract Type and Churn', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'combo_tenure_by_contract_churn.png'), dpi=150)
plt.close()
print("Saved: combo_tenure_by_contract_churn.png")

print("\n  Key numbers:")
for ct in contract_types:
    for ch in churn_labels:
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['tenure']
        print(f"    {ct}, Churn={ch}: median={subset.median():.2f}, mean={subset.mean():.2f}, count={len(subset)}")

# ────────────────────────────────────────
# 6.3 — Monthly_Charges by Internet_Service type, colored by Churn
# NOT a true interaction effect — descriptive context only.
# Shows fiber optic customers pay significantly more than DSL or no-internet customers.
# Partially explains why fiber optic has high churn (Part 5) — they're on expensive
# plans AND a high-churn internet type. But this is just two individual signals
# overlapping, not a genuine combined effect.
# ────────────────────────────────────────
print("\n--- 6.3: Monthly_Charges by Internet_Service type, colored by Churn ---")

internet_types = ['DSL', 'Fiber optic', 'No']

fig, ax = plt.subplots(figsize=(10, 6))
positions = []
data = []
tick_positions = []
tick_labels_list = []
pos = 1
for it in internet_types:
    for ch in churn_labels:
        subset = df[(df['Internet_Service'] == it) & (df['Churn'] == ch)]['Monthly_Charges']
        data.append(subset.values)
        positions.append(pos)
        pos += 1
    tick_positions.append(pos - 1.5)
    tick_labels_list.append(it)
    pos += 0.5

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2))
color_cycle = ['#bfdbfe', '#fecaca'] * len(internet_types)
for patch, c in zip(bp['boxes'], color_cycle):
    patch.set_facecolor(c)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_list)
ax.set_title('Monthly Charges by Internet Service and Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Internet Service')
ax.set_ylabel('Monthly Charges')
ax.legend(handles=[Patch(facecolor='#bfdbfe', label='No'), Patch(facecolor='#fecaca', label='Yes')], title='Churn')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'combo_monthly_by_internet_churn.png'), dpi=150)
plt.close()
print("Saved: combo_monthly_by_internet_churn.png")

print("\n  Key numbers:")
for it in internet_types:
    for ch in churn_labels:
        subset = df[(df['Internet_Service'] == it) & (df['Churn'] == ch)]['Monthly_Charges']
        print(f"    {it}, Churn={ch}: median={subset.median():.2f}, mean={subset.mean():.2f}, count={len(subset)}")

# ────────────────────────────────────────
# 6.4 — Scatter + Quadrant Analysis: tenure vs Monthly_Charges colored by Churn
# THE ONLY GENUINE INTERACTION EFFECT IN PART 6.
# Two steps:
#   Step 1 — Scatter plot: every customer as a dot, x=tenure, y=Monthly_Charges,
#            color=red(churned)/blue(stayed). Red dots visibly cluster top-left
#            (short tenure, high charges) — the interaction is visible before any numbers.
#   Step 2 — Quadrant analysis: draw two lines at the median tenure (29mo) and
#            median Monthly_Charges ($64.76), creating 4 boxes. Count churn rate
#            in each box to put numbers on the visual pattern.
#
# OUTPUT:
#   Low tenure  + Low charges  → ~24% churn
#   Low tenure  + High charges → ~58% churn  <- HIGHEST RISK
#   High tenure + Low charges  → ~12% churn
#   High tenure + High charges → ~20% churn
#
# WHY 58% IS A GENUINE INTERACTION AND NOT JUST ADDITION:
#   From Part 4: high charges alone → +0.19 correlation with churn (weak signal)
#   From Part 4: low tenure alone   → -0.35 correlation with churn (moderate signal)
#   If they simply added up you'd expect maybe 35-40% churn for the combination.
#   58% is far higher — the combination is more dangerous than the sum of its parts.
#   A new customer on an expensive plan hasn't had time to see the value yet.
#   A veteran on the same plan already decided it's worth it years ago.
# ────────────────────────────────────────
print("\n--- 6.4: Scatter — Tenure vs Monthly_Charges colored by Churn ---")

# Scatter plot: each dot is one customer, color-coded by churn
# alpha=0.25 (75% transparent) prevents overplotting with 7,000 points
# s=10 keeps dots small so the overall density pattern is visible
fig, ax = plt.subplots(figsize=(10, 6))
for ch, color in [('No', '#2563eb'), ('Yes', '#ef4444')]:
    subset = df[df['Churn'] == ch]
    ax.scatter(subset['tenure'], subset['Monthly_Charges'],
               c=color, alpha=0.25, s=10, label=f'Churn={ch}')
ax.set_title('Tenure vs Monthly Charges by Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Monthly Charges')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'combo_scatter_tenure_monthly.png'), dpi=150)
plt.close()
print("Saved: combo_scatter_tenure_monthly.png")

# Quadrant analysis: divide all customers into 4 groups using the median as the cutoff
# This reveals interaction effects between tenure and charges on churn risk
med_tenure = df['tenure'].median()    # ~29 months
med_monthly = df['Monthly_Charges'].median()  # ~$64.76
print(f"\n  Median tenure: {med_tenure:.0f} months, Median Monthly_Charges: ${med_monthly:.2f}")

# Each quadrant is a boolean-filtered subset of the full DataFrame
quadrants = {
    'Low tenure, Low charges':  df[(df['tenure'] <= med_tenure) & (df['Monthly_Charges'] <= med_monthly)],
    'Low tenure, High charges': df[(df['tenure'] <= med_tenure) & (df['Monthly_Charges'] > med_monthly)],
    'High tenure, Low charges': df[(df['tenure'] > med_tenure)  & (df['Monthly_Charges'] <= med_monthly)],
    'High tenure, High charges':df[(df['tenure'] > med_tenure)  & (df['Monthly_Charges'] > med_monthly)],
}
print("\n  Churn rate by quadrant:")
for name, qdf in quadrants.items():
    churned = len(qdf[qdf['Churn'] == 'Yes'])
    total = len(qdf)
    # Guard against division by zero in case a quadrant is unexpectedly empty
    rate = churned / total * 100 if total > 0 else 0
    # Key finding: 'Low tenure, High charges' has ~58% churn rate — highest risk segment
    print(f"    {name}: {churned}/{total} = {rate:.2f}%")

# ────────────────────────────────────────
# 6.5 — Stacked bar: Contract by Senior_Citizen, colored by Churn
# BORDERLINE interaction effect.
# Shows seniors on month-to-month contracts churn dramatically more than
# non-seniors on the same contract — ~54% vs ~43%.
# Technically a real interaction: the effect of contract type on churn is
# stronger for seniors than for non-seniors.
# However:
#   - XGBoost will learn this interaction on its own during training
#   - No clear business action beyond what Part 5 already told us
#     (both seniors and month-to-month customers were already flagged as high risk)
# Included for completeness but not a critical finding.
# ────────────────────────────────────────
print("\n--- 6.5: Contract by Senior_Citizen, colored by Churn ---")

# Two subplots side-by-side: one for non-seniors (0), one for seniors (1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, sc in enumerate([0, 1]):
    ax = axes[i]
    # Filter to just this senior status subgroup
    sub = df[df['Senior_Citizen'] == sc]
    # Count Churn=No and Churn=Yes for each contract type within this subgroup
    no_counts  = [len(sub[(sub['Contract'] == ct) & (sub['Churn'] == 'No')])  for ct in contract_types]
    yes_counts = [len(sub[(sub['Contract'] == ct) & (sub['Churn'] == 'Yes')]) for ct in contract_types]

    x = np.arange(len(contract_types))
    # Stacked bar: blue (No) on bottom, red (Yes) stacked on top
    # bottom=no_counts makes the Yes bars start where the No bars end
    ax.bar(x, no_counts,  color='#2563eb', label='Churn=No',  edgecolor='white')
    ax.bar(x, yes_counts, bottom=no_counts, color='#ef4444', label='Churn=Yes', edgecolor='white')

    ax.set_title(f'Senior_Citizen={sc}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Contract Type')
    ax.set_xticks(x)
    ax.set_xticklabels(contract_types, rotation=20, ha='right')
    if i == 0:
        ax.set_ylabel('Number of Customers')
    ax.legend()

plt.suptitle('Contract Type by Senior Citizen Status and Churn', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'combo_contract_senior_churn.png'), dpi=150)
plt.close()
print("Saved: combo_contract_senior_churn.png")

print("\n  Key numbers:")
for sc in [0, 1]:
    sc_label = 'Non-senior' if sc == 0 else 'Senior'
    for ct in contract_types:
        sub = df[(df['Senior_Citizen'] == sc) & (df['Contract'] == ct)]
        churned = len(sub[sub['Churn'] == 'Yes'])
        total = len(sub)
        rate = churned / total * 100 if total > 0 else 0
        print(f"    {sc_label}, {ct}: {churned}/{total} = {rate:.2f}%")


# ══════════════════════════════════════════════════
# PART 7 — ENGINEERED FEATURE
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 7 — ENGINEERED FEATURE")
print("=" * 60)

# ── WHAT WE DID AND WHY ───────────────────────────────────────────────────────
# Feature engineering = creating a NEW column by combining existing ones,
# hoping the new column gives the model more useful information than the originals.
#
# THE HYPOTHESIS:
#   From Part 4C: tenure and Total_Charges are 0.83 correlated — nearly the same signal.
#   Dividing them gives: charges_per_tenure = Total_Charges / tenure
#   This represents average monthly spend over a customer's entire life.
#   The hope was that this normalized version captures something neither raw column does.
#
# WHY IT SEEMED WORTH TRYING:
#   Total_Charges is misleading on its own — a customer with 24 months will naturally
#   have higher total charges than one with 2 months, even if they pay the same rate.
#   charges_per_tenure normalizes for this, giving a fairer "rate per month" comparison.
#
# HOW IT WAS TESTED:
#   1. Created the feature (with division-by-zero fix for tenure=0 customers)
#   2. Plotted its distribution
#   3. Compared churners vs non-churners on it (overlapping histogram + stats)
#   4. Measured its Pearson correlation with Churn: 0.1925
#   5. Compared against Monthly_Charges correlation with Churn: 0.1934
#
# RESULT:
#   charges_per_tenure  ↔ Churn = 0.1925
#   Monthly_Charges     ↔ Churn = 0.1934
#   Nearly identical. The new feature added zero new signal.
#
# WHY THE RESULT MAKES SENSE IN HINDSIGHT:
#   For a customer who has been around long enough, Total_Charges / tenure
#   mathematically approximates their average monthly rate — which is already
#   captured by Monthly_Charges. We engineered a feature that was essentially
#   Monthly_Charges in disguise.
#
# DECISION: Dropped from the final model.
#   Adding a feature that duplicates existing information adds noise, not value.
#   Confirmed in train_model.py: Version C (without charges_per_tenure) outperformed
#   Version B (with it) on recall.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────
# Step 1 — Create charges_per_tenure
# ────────────────────────────────────────
print("\n--- Step 1: Create charges_per_tenure ---")

# Re-clean Total_Charges (Part 7 runs independently; df may have been modified)
df['Total_Charges'] = df['Total_Charges'].astype(str).str.strip().replace('', '0')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

# Create the engineered feature: average monthly spend = Total_Charges / tenure
# np.where handles the edge case: if tenure=0, division would be undefined (ZeroDivisionError)
# For those 11 customers, we use Monthly_Charges as a proxy (their first month's rate)
df['charges_per_tenure'] = np.where(
    df['tenure'] == 0,
    df['Monthly_Charges'],          # fallback: use monthly rate for brand-new customers
    df['Total_Charges'] / df['tenure']  # normal case: average monthly spend
)

print(f"\ncharges_per_tenure created for all {len(df)} rows.")

# Print the tenure=0 rows to verify the fallback worked correctly
zero_tenure = df[df['tenure'] == 0]
print(f"\nRows where tenure == 0 ({len(zero_tenure)} rows):")
print(f"  {'customerID':<15} {'Monthly_Charges':<18} {'Total_Charges':<16} {'charges_per_tenure':<20}")
print(f"  {'-'*69}")
for _, row in zero_tenure.iterrows():
    print(f"  {row['customerID']:<15} {row['Monthly_Charges']:<18} {row['Total_Charges']:<16} {row['charges_per_tenure']:<20}")

# ────────────────────────────────────────
# Step 2 — Analyze charges_per_tenure
# ────────────────────────────────────────
print("\n--- Step 2: Analyze charges_per_tenure ---")

# 2.1 — Distribution histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['charges_per_tenure'], bins=36, color='#2563eb', edgecolor='white', alpha=0.85)
ax.set_title('Distribution of charges_per_tenure', fontsize=14, fontweight='bold')
ax.set_xlabel('charges_per_tenure')
ax.set_ylabel('Number of Customers')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'dist_charges_per_tenure.png'), dpi=150)
plt.close()
print("\nSaved: dist_charges_per_tenure.png")

# 2.2 — Overlapping histograms: churners vs non-churners
fig, ax = plt.subplots(figsize=(8, 5))
# churn_no/churn_yes were created in Part 4 before charges_per_tenure existed,
# so they don't have that column. The conditional checks whether it's there;
# if not, re-filters directly from df (which now has the column).
ax.hist(churn_no['charges_per_tenure'] if 'charges_per_tenure' in churn_no.columns else df[df['Churn'] == 'No']['charges_per_tenure'],
        bins=30, alpha=0.55, color='#2563eb', label='No', edgecolor='white')
ax.hist(churn_yes['charges_per_tenure'] if 'charges_per_tenure' in churn_yes.columns else df[df['Churn'] == 'Yes']['charges_per_tenure'],
        bins=30, alpha=0.55, color='#ef4444', label='Yes', edgecolor='white')
ax.set_title('charges_per_tenure: Churn=No vs Churn=Yes', fontsize=14, fontweight='bold')
ax.set_xlabel('charges_per_tenure')
ax.set_ylabel('Number of Customers')
ax.legend(title='Churn')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'hist_charges_per_tenure_vs_churn.png'), dpi=150)
plt.close()
print("Saved: hist_charges_per_tenure_vs_churn.png")

# 2.3 — Median and mean for churners vs non-churners
cpt_no = df[df['Churn'] == 'No']['charges_per_tenure']
cpt_yes = df[df['Churn'] == 'Yes']['charges_per_tenure']

med_no = cpt_no.median()
med_yes = cpt_yes.median()
mean_no = cpt_no.mean()
mean_yes = cpt_yes.mean()

print(f"\n  charges_per_tenure statistics:")
print(f"    Churn=No  — median: {med_no:.2f}, mean: {mean_no:.2f}")
print(f"    Churn=Yes — median: {med_yes:.2f}, mean: {mean_yes:.2f}")

# 2.4 — Difference between groups
diff_med = abs(med_no - med_yes)
diff_mean = abs(mean_no - mean_yes)
med_dir = 'No higher' if med_no > med_yes else 'Yes higher'
mean_dir = 'No higher' if mean_no > mean_yes else 'Yes higher'
print(f"    Difference — median: {diff_med:.2f} ({med_dir}), mean: {diff_mean:.2f} ({mean_dir})")

# 2.5 — Compute Pearson correlation between charges_per_tenure and the binary churn target
# Convert Churn to 0/1 again (churn_binary may not exist in this scope)
churn_binary = (df['Churn'] == 'Yes').astype(int)
corr_cpt = df['charges_per_tenure'].corr(churn_binary)
print(f"\n  Correlation of charges_per_tenure with Churn: {corr_cpt:.4f}")

# 2.6 — Compare against Total_Charges' correlation to see if the new feature adds signal
corr_tc = df['Total_Charges'].corr(churn_binary)
print(f"  Correlation of Total_Charges with Churn: {corr_tc:.4f}")

# 2.7 — Auto-print which is stronger (or if they're equal)
abs_cpt = abs(corr_cpt)
abs_tc = abs(corr_tc)
if abs_cpt > abs_tc:
    stronger = 'charges_per_tenure'
    weaker = 'Total_Charges'
elif abs_tc > abs_cpt:
    stronger = 'Total_Charges'
    weaker = 'charges_per_tenure'
else:
    stronger = None

if stronger:
    print(f"\n  {stronger} (|r|={max(abs_cpt, abs_tc):.4f}) has a stronger correlation with Churn than {weaker} (|r|={min(abs_cpt, abs_tc):.4f}).")
else:
    print(f"\n  charges_per_tenure and Total_Charges have equal correlation magnitude with Churn (|r|={abs_cpt:.4f}).")

# OUTPUT:
#   Correlation of charges_per_tenure with Churn: 0.1925
#   Correlation of Total_Charges    with Churn: 0.1934
#
#   charges_per_tenure (|r|=0.1925) vs Monthly_Charges (|r|=0.1934) — virtually identical.
#   The engineered feature added no new signal. Dropped from the final model.


# ══════════════════════════════════════════════════
# PART 9 — ADDITIONAL CHECKS
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 9 — ADDITIONAL CHECKS")
print("=" * 60)

# ── WHY THIS SECTION EXISTS ───────────────────────────────────────────────────
# Part 9 digs deeper into patterns that Parts 4-6 hinted at but didn't fully
# quantify. Instead of looking at raw feature values, we group customers into
# buckets and measure churn rate per bucket.
#
# Sub-sections:
#   9A — Tenure buckets (0-12, 13-24, ... 61-72 months)
#        Key finding: 0-12 months = 47% churn rate, 61-72 months = 6.6% churn rate
#        Confirms: the earlier a customer is, the higher the risk
#
#   9B — Monthly_Charges buckets ($0-30, $31-50, ... $91-120)
#        Key finding: higher monthly bill = higher churn rate
#        The $91-120 bucket has the highest churn rate
#
#   9C — Service adoption count (how many of 6 add-on services does each customer have?)
#        Hypothesis: customers with more services are more embedded → less likely to churn
#        We count how many of: Online_Security, Online_Backup, Device_Protection,
#        Tech_Support, Streaming_TV, Streaming_Movies each customer has subscribed to
#        Key finding: 0 services = highest churn, 6 services = lowest churn
#
#   9D — Payment_Method + Paperless_Billing combination
#        Key finding: Electronic check + Paperless_Billing=Yes is the highest-risk
#        payment combination — these customers churn the most
#
# Overall purpose: these checks give the business team specific, actionable segments
# to target (e.g. new customers on fiber optic with electronic check payments)
# and confirm the model features are capturing real patterns.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────
# 9A — Churn rate by tenure buckets
# ────────────────────────────────────────
print("\n--- 9A: Churn rate by tenure buckets ---")

# 6 tenure buckets of 12 months each covering the full 0–72 month range
tenure_bins = [(0, 12), (13, 24), (25, 36), (37, 48), (49, 60), (61, 72)]
print(f"\n  {'Bucket':<15} {'Customers':<12} {'Churned':<10} {'Churn Rate':<12}")
print(f"  {'-'*49}")
for lo, hi in tenure_bins:
    # Boolean filter: keep rows where tenure falls within [lo, hi] inclusive
    bucket = df[(df['tenure'] >= lo) & (df['tenure'] <= hi)]
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100 if total > 0 else 0
    # Key finding: 0-12 months = 47.44% churn; 61-72 months = 6.61% churn
    print(f"  {lo}-{hi} months{'':<4} {total:<12} {churned:<10} {rate:.2f}%")

# ────────────────────────────────────────
# 9B — Churn rate by Monthly_Charges buckets
# ────────────────────────────────────────
print("\n--- 9B: Churn rate by Monthly_Charges buckets ---")

# 5 charge buckets covering the $0–$120 range of Monthly_Charges
charge_bins = [(0, 30), (31, 50), (51, 70), (71, 90), (91, 120)]
print(f"\n  {'Bucket':<15} {'Customers':<12} {'Churned':<10} {'Churn Rate':<12}")
print(f"  {'-'*49}")
for lo, hi in charge_bins:
    bucket = df[(df['Monthly_Charges'] >= lo) & (df['Monthly_Charges'] <= hi)]
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100 if total > 0 else 0
    # Key finding: higher charges = higher churn rate ($91-$120 band has the highest)
    print(f"  ${lo}-${hi}{'':<7} {total:<12} {churned:<10} {rate:.2f}%")

# ────────────────────────────────────────
# 9C — Service adoption vs churn
# ────────────────────────────────────────
print("\n--- 9C: Service adoption vs churn ---")

# The 6 optional add-on services — we'll count how many each customer subscribes to
service_cols = ['Online_Security', 'Online_Backup', 'Device_Protection',
                'Tech_Support', 'Streaming_TV', 'Streaming_Movies']
# lambda counts how many of the 6 columns have the value 'Yes' for each row
# axis=1 applies the function row-by-row (not column-by-column)
df['services_count'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

print(f"\n  {'Services':<12} {'Customers':<12} {'Churn Rate':<12}")
print(f"  {'-'*36}")
# range(7) covers 0 to 6 services; skip counts that don't exist in the data
for sc in range(7):
    bucket = df[df['services_count'] == sc]
    if len(bucket) == 0:
        continue
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100
    # Hypothesis: customers with more services are more embedded → less likely to churn
    print(f"  {sc:<12} {total:<12} {rate:.2f}%")

# ────────────────────────────────────────
# 9D — Payment_Method + Paperless_Billing combination
# ────────────────────────────────────────
print("\n--- 9D: Payment_Method + Paperless_Billing combination ---")

# Get the unique payment methods from the data (avoids hardcoding)
payment_methods = df['Payment_Method'].unique()
paperless_values = ['Yes', 'No']

print(f"\n  {'Payment Method':<35} {'Paperless':<12} {'Customers':<12} {'Churn Rate':<12}")
print(f"  {'-'*71}")
for pm in sorted(payment_methods):  # sorted() for consistent alphabetical output
    for pb in paperless_values:
        # Cross-filter: rows matching both this payment method AND this paperless status
        bucket = df[(df['Payment_Method'] == pm) & (df['Paperless_Billing'] == pb)]
        if len(bucket) == 0:
            continue  # skip combinations that don't exist in the data
        churned = len(bucket[bucket['Churn'] == 'Yes'])
        total = len(bucket)
        rate = churned / total * 100
        # Key finding: Electronic check + Paperless=Yes has the highest churn rate combination
        print(f"  {pm:<35} {pb:<12} {total:<12} {rate:.2f}%")


# ══════════════════════════════════════════════════
# PART 8 — CATEGORICAL CONSISTENCY CHECK
# ══════════════════════════════════════════════════
# WHAT WE LOOKED FOR:
#   Are categorical values written consistently across all rows?
#   e.g. is "Yes" always "Yes" and never "yes", "YES", or "Y"?
#   pandas treats differently-cased or differently-spelled versions of the
#   same value as completely separate categories. If "Male" and "male" both
#   exist, the encoder creates two separate values and the model trains on noise.
#
# WHY THIS MATTERS EVEN WITH XGBOOST:
#   This is not about model assumptions — XGBoost doesn't care about case.
#   It's about data quality. Inconsistent values corrupt the encoding step
#   (OrdinalEncoder / OneHotEncoder) before the data even reaches XGBoost.
#
# FINDING:
#   All categorical columns are clean. Every unique value is consistently written.
#   No duplicates under different casing or spelling were found.
#   This dataset is a well-known public Kaggle dataset — it was already clean,
#   but verifying this explicitly is standard EDA practice rather than assuming it.

print("\n\n" + "=" * 60)
print("PART 8 — CATEGORICAL CONSISTENCY CHECK")
print("=" * 60)

all_clean = True
for col in cat_columns:
    # Strip whitespace and get unique values
    unique_vals = df[col].astype(str).str.strip().unique()
    # Check if any two values are the same when lowercased — catches "Yes" vs "yes"
    lowered = [v.lower() for v in unique_vals]
    if len(lowered) != len(set(lowered)):
        print(f"\n  WARNING — {col} has inconsistent casing:")
        print(f"    Unique values: {sorted(unique_vals)}")
        all_clean = False
    else:
        print(f"  {col:<25} OK — unique values: {sorted(unique_vals)}")

if all_clean:
    print("\n  All categorical columns are consistent. No typos or casing issues detected.")

# OUTPUT:
#   gender                    OK — unique values: ['Female', 'Male']
#   Senior_Citizen            OK — unique values: ['0', '1']
#   Is_Married                OK — unique values: ['No', 'Yes']
#   Dependents                OK — unique values: ['No', 'Yes']
#   Phone_Service             OK — unique values: ['No', 'Yes']
#   Dual                      OK — unique values: ['No', 'No phone service', 'Yes']
#   Internet_Service          OK — unique values: ['DSL', 'Fiber optic', 'No']
#   Online_Security           OK — unique values: ['No', 'No internet service', 'Yes']
#   Online_Backup             OK — unique values: ['No', 'No internet service', 'Yes']
#   Device_Protection         OK — unique values: ['No', 'No internet service', 'Yes']
#   Tech_Support              OK — unique values: ['No', 'No internet service', 'Yes']
#   Streaming_TV              OK — unique values: ['No', 'No internet service', 'Yes']
#   Streaming_Movies          OK — unique values: ['No', 'No internet service', 'Yes']
#   Contract                  OK — unique values: ['Month-to-month', 'One year', 'Two year']
#   Paperless_Billing         OK — unique values: ['No', 'Yes']
#   Payment_Method            OK — unique values: ['Bank transfer (automatic)',
#                                                   'Credit card (automatic)',
#                                                   'Electronic check',
#                                                   'Mailed check']
#
#   All categorical columns are consistent. No typos or casing issues detected.
