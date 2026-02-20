import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'eda_plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ══════════════════════════════════════════════════
# PART 1 — BASIC DATA UNDERSTANDING
# ══════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("=" * 60)
print("PART 1 — BASIC DATA UNDERSTANDING")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nFirst 5 rows:")
print(df.head().to_string())

print(f"\n{'Column':<25} {'Dtype':<15}")
print("-" * 40)
for col in df.columns:
    print(f"{col:<25} {str(df[col].dtype):<15}")

# Basic statistics for numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nBasic statistics for numeric columns ({', '.join(numeric_cols)}):")
print(df[numeric_cols].describe().to_string())
# Note: Total_Charges is not included here because pandas reads it as object, not numeric.


# ══════════════════════════════════════════════════
# PART 2 — DATA QUALITY
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 2 — DATA QUALITY")
print("=" * 60)

# Null/missing values per column
print("\nNull values per column:")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("  No standard null values detected in any column.")
else:
    print(null_counts[null_counts > 0].to_string())

# Check for blank/whitespace strings across ALL columns
print("\nBlank/whitespace strings per column:")
blank_found = False
for col in df.columns:
    blank_count = (df[col].astype(str).str.strip() == '').sum()
    if blank_count > 0:
        print(f"  {col}: {blank_count}")
        blank_found = True
if not blank_found:
    print("  No blank/whitespace strings detected in any column.")

# Detail rows with blank Total_Charges
blank_tc = df[df['Total_Charges'].astype(str).str.strip() == '']

# Print full details of those rows so we can see what kind of customers they are
if len(blank_tc) > 0:
    print(f"\nFull details of rows with blank Total_Charges:")
    print(blank_tc.to_string())

# Duplicate rows
dup_count = df.duplicated().sum()
print(f"\nCompletely duplicate rows: {dup_count}")

# Unique values per column
print(f"\nUnique values per column:")
print(f"{'Column':<25} {'Unique Count':<15}")
print("-" * 40)
for col in df.columns:
    print(f"{col:<25} {df[col].nunique():<15}")


# ══════════════════════════════════════════════════
# PART 3 — TARGET VARIABLE
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 3 — TARGET VARIABLE")
print("=" * 60)

churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn column value counts:")
for val in churn_counts.index:
    print(f"  {val}: {churn_counts[val]}  ({churn_pct[val]:.2f}%)")

# Bar chart: count of each Churn value with exact numbers on top
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ['No', 'Yes'],
    [churn_counts['No'], churn_counts['Yes']],
    color=['#2563eb', '#ef4444'], edgecolor='white', width=0.5
)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 50,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('Churn Count', fontsize=14, fontweight='bold')
ax.set_xlabel('Churn')
ax.set_ylabel('Number of Customers')
ax.set_ylim(0, max(churn_counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_bar.png'), dpi=150)
plt.close()
print(f"\nSaved: target_bar.png")

# Pie chart: percentage split
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    [churn_counts['No'], churn_counts['Yes']],
    labels=['No', 'Yes'],
    autopct='%1.1f%%',
    colors=['#2563eb', '#ef4444'],
    startangle=90,
    textprops={'fontsize': 12}
)
ax.set_title('Churn Percentage Split', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'target_pie.png'), dpi=150)
plt.close()
print(f"Saved: target_pie.png")

# The majority class is "No" — this is the baseline any model must beat
majority_pct = churn_pct.max()
print(f"\nA model that always predicts the majority class would achieve {majority_pct:.1f}% accuracy with 0% Recall.")
print(f"Our model must be evaluated against this baseline.")


# ══════════════════════════════════════════════════
# PART 4 — NUMERICAL FEATURES
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 4 — NUMERICAL FEATURES")
print("=" * 60)

# Clean Total_Charges for numeric analysis
df['Total_Charges'] = df['Total_Charges'].str.strip().replace('', '0')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
print(f"\nAfter converting Total_Charges to numeric: {len(df)} rows remain (no rows dropped)")

# ────────────────────────────────────────
# 4A — Distributions
# ────────────────────────────────────────
print("\n--- 4A: Distributions ---")

num_features = ['tenure', 'Monthly_Charges', 'Total_Charges']

for feat in num_features:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[feat], bins=36, color='#2563eb', edgecolor='white', alpha=0.85)
    ax.set_title(f'Distribution of {feat}', fontsize=14, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Number of Customers')
    plt.tight_layout()
    fname = f'dist_{feat.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")

print("\n4A Observations:")
print("  tenure: two concentrations are visible — one at the lowest values and one at the highest, with a flatter middle section.")
print("  Monthly_Charges: a tall cluster near ~$20 and a broad, roughly uniform spread from ~$40 to ~$110.")
print("  Total_Charges: most values are concentrated near the low end, tapering off gradually toward higher values.")

# ────────────────────────────────────────
# 4B — Numerical features vs Churn
# ────────────────────────────────────────
print("\n--- 4B: Numerical Features vs Churn ---")

churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']

for feat in num_features:
    # Box plot: side by side by churn status
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [churn_no[feat].values, churn_yes[feat].values],
        tick_labels=['No', 'Yes'], patch_artist=True,
        medianprops=dict(color='red', linewidth=2)
    )
    bp['boxes'][0].set_facecolor('#bfdbfe')
    bp['boxes'][1].set_facecolor('#fecaca')
    ax.set_title(f'{feat} by Churn Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Churn')
    ax.set_ylabel(feat)
    plt.tight_layout()
    fname_box = f'box_{feat.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname_box), dpi=150)
    plt.close()

    # Overlapping histogram: churners vs non-churners
    fig, ax = plt.subplots(figsize=(8, 5))
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

    # Statistics
    med_no = churn_no[feat].median()
    med_yes = churn_yes[feat].median()
    mean_no = churn_no[feat].mean()
    mean_yes = churn_yes[feat].mean()

    print(f"\n  {feat}:")
    print(f"    Churn=No  — median: {med_no:.2f}, mean: {mean_no:.2f}")
    print(f"    Churn=Yes — median: {med_yes:.2f}, mean: {mean_yes:.2f}")
    print(f"    Difference — median: {abs(med_no - med_yes):.2f} ({'No higher' if med_no > med_yes else 'Yes higher'}), mean: {abs(mean_no - mean_yes):.2f} ({'No higher' if mean_no > mean_yes else 'Yes higher'})")
    print(f"    Saved: {fname_box}, {fname_hist}")

print("\n4B Observations:")
print("  tenure: Churn=Yes group has a median of 10 and Churn=No has 38 — a gap of 28 months.")
print("  Monthly_Charges: Churn=Yes group has a higher median than Churn=No.")
print("  Total_Charges: Churn=Yes group has a lower median than Churn=No.")
print("  Total_Charges combines both tenure duration and monthly rate, so it reflects both factors at once.")

# ────────────────────────────────────────
# 4C — Correlation
# ────────────────────────────────────────
print("\n--- 4C: Correlation ---")

df_corr = df[['tenure', 'Monthly_Charges', 'Total_Charges']].copy()
df_corr['Churn'] = (df['Churn'] == 'Yes').astype(int)

corr_matrix = df_corr.corr()

print(f"\nCorrelation matrix:")
for col in corr_matrix.columns:
    vals = '  '.join(f"{corr_matrix.loc[col, c]:>7.4f}" for c in corr_matrix.columns)
    print(f"  {col:<20} {vals}")

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


# ══════════════════════════════════════════════════
# PART 5 — CATEGORICAL FEATURES
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 5 — CATEGORICAL FEATURES")
print("=" * 60)

cat_columns = [
    'gender', 'Senior_Citizen', 'Is_Married', 'Dependents',
    'Phone_Service', 'Dual', 'Internet_Service',
    'Online_Security', 'Online_Backup', 'Device_Protection',
    'Tech_Support', 'Streaming_TV', 'Streaming_Movies',
    'Contract', 'Paperless_Billing', 'Payment_Method',
]

churn_rate_ranges = {}  # column -> max churn rate difference across its values

for col in cat_columns:
    print(f"\n{'-' * 50}")
    print(f"  {col}")
    print(f"{'-' * 50}")

    # Step A: Value counts
    print(f"\n  Value counts:")
    vc = df[col].value_counts()
    for val in vc.index:
        print(f"    {val}: {vc[val]}")

    # Step B: Churn rate per category value
    print(f"\n  Churn rate per value:")
    churn_rates = {}
    for val in vc.index:
        subset = df[df[col] == val]
        churned = subset[subset['Churn'] == 'Yes']
        rate = len(churned) / len(subset) * 100
        churn_rates[val] = rate
        print(f"    {val}: {len(churned)}/{len(subset)} = {rate:.2f}%")

    max_rate = max(churn_rates.values())
    min_rate = min(churn_rates.values())
    churn_rate_ranges[col] = max_rate - min_rate

    # Step C: Grouped bar chart — churned vs not churned side by side
    categories = list(vc.index)
    no_counts = [len(df[(df[col] == val) & (df['Churn'] == 'No')]) for val in categories]
    yes_counts = [len(df[(df[col] == val) & (df['Churn'] == 'Yes')]) for val in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 5))
    bars_no = ax.bar(x - width / 2, no_counts, width, label='No', color='#2563eb', edgecolor='white')
    bars_yes = ax.bar(x + width / 2, yes_counts, width, label='Yes', color='#ef4444', edgecolor='white')

    ax.set_title(f'{col}: Churned vs Not Churned', fontsize=14, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Number of Customers')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in categories], rotation=45 if len(categories) > 3 else 0, ha='right' if len(categories) > 3 else 'center')
    ax.legend(title='Churn')
    plt.tight_layout()
    fname = f'cat_{col.lower()}.png'
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
    plt.close()
    print(f"\n  Saved: {fname}")

# Ranked list by churn rate difference
print(f"\n\n{'=' * 60}")
print("CATEGORICAL FEATURES RANKED BY CHURN RATE DIFFERENCE")
print("=" * 60)
print(f"\n{'Rank':<6} {'Feature':<25} {'Max Difference':<15}")
print("-" * 46)

ranked = sorted(churn_rate_ranges.items(), key=lambda x: x[1], reverse=True)
for i, (col, diff) in enumerate(ranked, 1):
    print(f"{i:<6} {col:<25} {diff:.2f} pp")


# ══════════════════════════════════════════════════
# PART 6 — FEATURE COMBINATIONS
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 6 — FEATURE COMBINATIONS")
print("=" * 60)

# ────────────────────────────────────────
# 6.1 — Monthly_Charges by Contract type, colored by Churn
# ────────────────────────────────────────
print("\n--- 6.1: Monthly_Charges by Contract type, colored by Churn ---")

contract_types = ['Month-to-month', 'One year', 'Two year']
churn_labels = ['No', 'Yes']
colors = {'No': '#2563eb', 'Yes': '#ef4444'}

fig, ax = plt.subplots(figsize=(10, 6))
positions = []
data = []
tick_positions = []
tick_labels_list = []
pos = 1
for ct in contract_types:
    for ch in churn_labels:
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['Monthly_Charges']
        data.append(subset.values)
        positions.append(pos)
        pos += 1
    tick_positions.append(pos - 1.5)
    tick_labels_list.append(ct)
    pos += 0.5

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2))
color_cycle = ['#bfdbfe', '#fecaca'] * len(contract_types)
for patch, c in zip(bp['boxes'], color_cycle):
    patch.set_facecolor(c)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_list)
ax.set_title('Monthly Charges by Contract Type and Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Contract Type')
ax.set_ylabel('Monthly Charges')
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
# ────────────────────────────────────────
print("\n--- 6.2: Tenure by Contract type, colored by Churn ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for i, ct in enumerate(contract_types):
    ax = axes[i]
    for ch in churn_labels:
        subset = df[(df['Contract'] == ct) & (df['Churn'] == ch)]['tenure']
        ax.hist(subset, bins=20, alpha=0.55, color=colors[ch], label=f'Churn={ch}', edgecolor='white')
    ax.set_title(f'{ct}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tenure (months)')
    if i == 0:
        ax.set_ylabel('Number of Customers')
    ax.legend()
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
# 6.4 — Scatter: tenure vs Monthly_Charges colored by Churn
# ────────────────────────────────────────
print("\n--- 6.4: Scatter — Tenure vs Monthly_Charges colored by Churn ---")

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

# Quadrant analysis: split at median tenure and median monthly charges
med_tenure = df['tenure'].median()
med_monthly = df['Monthly_Charges'].median()
print(f"\n  Median tenure: {med_tenure:.0f} months, Median Monthly_Charges: ${med_monthly:.2f}")

quadrants = {
    'Low tenure, Low charges': df[(df['tenure'] <= med_tenure) & (df['Monthly_Charges'] <= med_monthly)],
    'Low tenure, High charges': df[(df['tenure'] <= med_tenure) & (df['Monthly_Charges'] > med_monthly)],
    'High tenure, Low charges': df[(df['tenure'] > med_tenure) & (df['Monthly_Charges'] <= med_monthly)],
    'High tenure, High charges': df[(df['tenure'] > med_tenure) & (df['Monthly_Charges'] > med_monthly)],
}
print("\n  Churn rate by quadrant:")
for name, qdf in quadrants.items():
    churned = len(qdf[qdf['Churn'] == 'Yes'])
    total = len(qdf)
    rate = churned / total * 100 if total > 0 else 0
    print(f"    {name}: {churned}/{total} = {rate:.2f}%")

# ────────────────────────────────────────
# 6.5 — Stacked bar: Contract by Senior_Citizen, colored by Churn
# ────────────────────────────────────────
print("\n--- 6.5: Contract by Senior_Citizen, colored by Churn ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, sc in enumerate([0, 1]):
    ax = axes[i]
    sub = df[df['Senior_Citizen'] == sc]
    no_counts = [len(sub[(sub['Contract'] == ct) & (sub['Churn'] == 'No')]) for ct in contract_types]
    yes_counts = [len(sub[(sub['Contract'] == ct) & (sub['Churn'] == 'Yes')]) for ct in contract_types]

    x = np.arange(len(contract_types))
    ax.bar(x, no_counts, color='#2563eb', label='Churn=No', edgecolor='white')
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

# ────────────────────────────────────────
# Step 1 — Create charges_per_tenure
# ────────────────────────────────────────
print("\n--- Step 1: Create charges_per_tenure ---")

df['Total_Charges'] = df['Total_Charges'].astype(str).str.strip().replace('', '0')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

df['charges_per_tenure'] = np.where(
    df['tenure'] == 0,
    df['Monthly_Charges'],
    df['Total_Charges'] / df['tenure']
)

print(f"\ncharges_per_tenure created for all {len(df)} rows.")

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

# 2.5 — Correlation with Churn
churn_binary = (df['Churn'] == 'Yes').astype(int)
corr_cpt = df['charges_per_tenure'].corr(churn_binary)
print(f"\n  Correlation of charges_per_tenure with Churn: {corr_cpt:.4f}")

# 2.6 — Correlation of Total_Charges with Churn for comparison
corr_tc = df['Total_Charges'].corr(churn_binary)
print(f"  Correlation of Total_Charges with Churn: {corr_tc:.4f}")

# 2.7 — Neutral comparison statement
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


# ══════════════════════════════════════════════════
# PART 9 — ADDITIONAL CHECKS
# ══════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 9 — ADDITIONAL CHECKS")
print("=" * 60)

# ────────────────────────────────────────
# 9A — Churn rate by tenure buckets
# ────────────────────────────────────────
print("\n--- 9A: Churn rate by tenure buckets ---")

tenure_bins = [(0, 12), (13, 24), (25, 36), (37, 48), (49, 60), (61, 72)]
print(f"\n  {'Bucket':<15} {'Customers':<12} {'Churned':<10} {'Churn Rate':<12}")
print(f"  {'-'*49}")
for lo, hi in tenure_bins:
    bucket = df[(df['tenure'] >= lo) & (df['tenure'] <= hi)]
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100 if total > 0 else 0
    print(f"  {lo}-{hi} months{'':<4} {total:<12} {churned:<10} {rate:.2f}%")

# ────────────────────────────────────────
# 9B — Churn rate by Monthly_Charges buckets
# ────────────────────────────────────────
print("\n--- 9B: Churn rate by Monthly_Charges buckets ---")

charge_bins = [(0, 30), (31, 50), (51, 70), (71, 90), (91, 120)]
print(f"\n  {'Bucket':<15} {'Customers':<12} {'Churned':<10} {'Churn Rate':<12}")
print(f"  {'-'*49}")
for lo, hi in charge_bins:
    bucket = df[(df['Monthly_Charges'] >= lo) & (df['Monthly_Charges'] <= hi)]
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100 if total > 0 else 0
    print(f"  ${lo}-${hi}{'':<7} {total:<12} {churned:<10} {rate:.2f}%")

# ────────────────────────────────────────
# 9C — Service adoption vs churn
# ────────────────────────────────────────
print("\n--- 9C: Service adoption vs churn ---")

service_cols = ['Online_Security', 'Online_Backup', 'Device_Protection',
                'Tech_Support', 'Streaming_TV', 'Streaming_Movies']
df['services_count'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

print(f"\n  {'Services':<12} {'Customers':<12} {'Churn Rate':<12}")
print(f"  {'-'*36}")
for sc in range(7):
    bucket = df[df['services_count'] == sc]
    if len(bucket) == 0:
        continue
    churned = len(bucket[bucket['Churn'] == 'Yes'])
    total = len(bucket)
    rate = churned / total * 100
    print(f"  {sc:<12} {total:<12} {rate:.2f}%")

# ────────────────────────────────────────
# 9D — Payment_Method + Paperless_Billing combination
# ────────────────────────────────────────
print("\n--- 9D: Payment_Method + Paperless_Billing combination ---")

payment_methods = df['Payment_Method'].unique()
paperless_values = ['Yes', 'No']

print(f"\n  {'Payment Method':<35} {'Paperless':<12} {'Customers':<12} {'Churn Rate':<12}")
print(f"  {'-'*71}")
for pm in sorted(payment_methods):
    for pb in paperless_values:
        bucket = df[(df['Payment_Method'] == pm) & (df['Paperless_Billing'] == pb)]
        if len(bucket) == 0:
            continue
        churned = len(bucket[bucket['Churn'] == 'Yes'])
        total = len(bucket)
        rate = churned / total * 100
        print(f"  {pm:<35} {pb:<12} {total:<12} {rate:.2f}%")
