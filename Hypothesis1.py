import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Output folder for plots
os.makedirs("Hypothesis1_Plots", exist_ok=True)


# LOCAL DATASET
# Loading and Cleaning
washington_dataset = pd.read_csv("Washington_DB.csv")
df_clean = washington_dataset.dropna(subset=['IndustryType', 'InformationType'])

# Contingency table (raw counts)
contingency_raw = pd.crosstab(df_clean['IndustryType'], df_clean['InformationType'])


# Chi-Square Test - How far the actual distribution of leaked information types across industries deviates from what we would expect if there were no relationship between the two?
chi2, p, dof, expected = chi2_contingency(contingency_raw)
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"p-value: {p:.12f}")  # extended precision

# Cramér’s V - How strongly is the type of information leaked associated with the industry sector?
n = contingency_raw.to_numpy().sum()
min_dim = min(contingency_raw.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
print(f"Cramér’s V: {cramers_v:.4f}")

# Normalized contingency table
contingency_normalized = pd.crosstab(
    df_clean['IndustryType'],
    df_clean['InformationType'],
    normalize='index'
)

# Renaming long labels
info_type_renames = {
    'Driver\'s License or Washington ID Card Number': 'Driver\'s License',
    'Email Address and Password/Security Question Answers': 'Email and Password',
    'Financial & Banking Information': 'Financial/Banking Information',
    'Health Insurance Policy or ID Number': 'Health Insurance or ID',
    'Unique Private Key (e.g. used to authenticate or sign an electronic record)': 'Unique Private Key',
    'Username and Password/Security Question Answers': 'Username and Password',
    'Full Date of Birth': 'Date of Birth',
    'Social Security Number': 'SSN',
}
industry_renames = {
    'Non-Profit/Charity': 'Non-Profit'
}
contingency_normalized = contingency_normalized.rename(columns=info_type_renames)
contingency_normalized = contingency_normalized.rename(index=industry_renames)

# "Other" column to the end
cols = list(contingency_normalized.columns)
if 'Other' in cols:
    cols.append(cols.pop(cols.index('Other')))
contingency_normalized = contingency_normalized[cols]


# Normalized heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    contingency_normalized,
    annot=True,
    fmt=".2f",
    cmap="BuPu",
    cbar=False
)
plt.title(
    "Share of Leaked Information Types per Industry",
    fontsize=16,
    fontweight='bold'
)
plt.ylabel("Industry Type", fontsize=12, fontweight='bold')
plt.xlabel("Information Type", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Hypothesis1_Plots/heatmap.png")
plt.show()


#GLOBAL DATASET
# Load and clean
global_dataset = pd.read_csv("Kaggle_DB_updated.csv")
global_dataset.columns = global_dataset.columns.str.strip()  # remove extra spaces
global_clean = global_dataset.dropna(subset=['sector', 'data sensitivity'])

# Contingency table (raw counts)
contingency_raw_global = pd.crosstab(global_clean['data sensitivity'], global_clean['sector'])

# Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_raw_global)
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"p-value: {p:.12f}")

# Cramér’s V
n = contingency_raw_global.to_numpy().sum()
min_dim = min(contingency_raw_global.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
print(f"Cramér’s V: {cramers_v:.4f}")

# Normalized heatmap
contingency_normalized = pd.crosstab(
    global_clean['data sensitivity'],
    global_clean['sector'],
    normalize='index'
)

# Optional: create folder if needed
os.makedirs("Hypothesis1_Plots", exist_ok=True)

plt.figure(figsize=(14, 8))
sns.heatmap(
    contingency_normalized,
    annot=True,
    fmt=".2f",
    cmap="BuPu",
    cbar=False  # Removes colorbar
)
plt.title("Share of Sector Distribution per Data Sensitivity Level", fontsize=16, fontweight='bold')
plt.ylabel("Data Sensitivity", fontsize=12, fontweight='bold')
plt.xlabel("Sector", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Hypothesis1_Plots/heatmap_global_flipped.png")
plt.show()



for sensitivity_level, row in contingency_normalized.iterrows():
    plt.figure(figsize=(10, 4))
    row.plot(kind='bar', color='mediumpurple', edgecolor='black')

    plt.title(f"Sector Distribution for Data Sensitivity Level {sensitivity_level}", fontsize=14, fontweight='bold')
    plt.xlabel("Sector", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Hypothesis1_Plots/barplot_sensitivity_{sensitivity_level}.png")
    plt.show()
