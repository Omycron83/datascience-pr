import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import os


df = pd.read_csv("Kaggle_DB_updated.csv")

# Replace all exact matches of 'financial' with 'finance' in the 'sector' column
df['sector'] = df['sector'].replace('financial', 'finance')
df['sector'] = df['sector'].astype(str).apply(lambda x: 'government' if 'government' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'military' if 'military' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'misc' if 'misc' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'tech' if 'tech' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'web' if 'web' in x else x)

df.to_csv("Kaggle_DB_updated.csv", index=False)

print(df)

# Overview of the dataset
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("Column names:")
print(df.columns.tolist())

print("Data types of each column:")
print(df.dtypes)

print("Summary statistics:")
print(df.describe(include='all'))

# Sum total breaches per sector
total_breaches_by_sector = df['sector'].value_counts().sort_values(ascending=False)
print("Total breaches per sector: \n", total_breaches_by_sector)

# Bar plot of total breaches per sector
plt.figure(figsize=(8, 6))
total_breaches_by_sector.plot(kind='bar', color='steelblue')
plt.title("Total Number of Breaches per Sector - Global Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of Breaches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("breach_bar_plot.png")
plt.show()

# Convert 'records lost' to numeric (remove commas if present), set errors='coerce' to handle non-numeric values
df['records_lost_numeric'] = pd.to_numeric(df['records lost'].astype(str).str.replace(',', ''), errors='coerce')

# Sum of records lost per sector
record_lost_by_sector = df.groupby('sector')['records_lost_numeric'].sum().sort_values(ascending=False)
print("Record lost by sector: \n", record_lost_by_sector)

# Bar plot of total records lost per sector
plt.figure(figsize=(8, 6))
record_lost_by_sector.plot(kind='bar', color='crimson')
plt.title("Total Records Lost per Sector - Global Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of Records Lost")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("record_lost_by_sector.png")
plt.show()


# Boxplot: Records Lost per Breach by Sector (log scale, mean overlay)
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='sector',
    y='records_lost_numeric',
    data=df,
    color='skyblue',
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
)
plt.yscale('log')  # Set y-axis to log scale
plt.title("Records Lost per Breach by Sector - Global Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of Records Lost per Breach (log scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig("records_lost_per_breach_boxplot_log.png")
plt.show()# Calculate mean and standard error for records lost per sector
mean_records_by_sector = df.groupby('sector')['records_lost_numeric'].agg(['mean', 'count', 'std'])
mean_records_by_sector['se'] = mean_records_by_sector['std'] / np.sqrt(mean_records_by_sector['count'])
print(mean_records_by_sector)

# Prepare the data for ANOVA: a list of arrays, one for each sector
sector_types = df['sector'].unique()
anova_groups = [df.loc[df['sector'] == sector, 'records_lost_numeric'].dropna() for sector in sector_types]

# Perform one-way ANOVA
anova_stat, anova_p = f_oneway(*anova_groups)
print(f"\nOne-way ANOVA result: F = {anova_stat:.4f}, p = {anova_p:.4g}")

if anova_p < 0.05:
    print("There is a significant difference between at least two sectors.")
else:
    print("No significant difference found between sectors.")


# multiple t-tests (independent samples, Welch’s t-test) with Bonferroni correction
sector_types = df['sector'].unique()
pairs = list(combinations(sector_types, 2))
ttest_pvals = []

for a, b in pairs:
    group_a = df.loc[df['sector'] == a, 'records_lost_numeric'].dropna()
    group_b = df.loc[df['sector'] == b, 'records_lost_numeric'].dropna()
    if len(group_a) > 1 and len(group_b) > 1:
        stat, p = ttest_ind(group_a, group_b, equal_var=False)
        ttest_pvals.append(p)
    else:
        ttest_pvals.append(np.nan)

# Bonferroni correction
reject, pvals_bonf, _, _ = multipletests(ttest_pvals, method='bonferroni')

print("\nPairwise t-test results (Bonferroni corrected):")
for idx, (a, b) in enumerate(pairs):
    print(f"{a} vs {b}: p = {pvals_bonf[idx]:.4g} {'*' if reject[idx] else ''}")