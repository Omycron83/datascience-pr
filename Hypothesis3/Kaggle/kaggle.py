import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import os


df = pd.read_csv("Kaggle_DB.csv")

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

# Remove the 2nd row from the dataframe (index 1 in Python)
df = df.drop(df.index[1])
df.to_csv("Kaggle_remove1.csv", index=False)
print(df.head())

# Calculate total number of breaches for each sector
breach_count_by_sector = df['sector'].value_counts().reset_index()
breach_count_by_sector.columns = ['sector', 'breach_count']
print(breach_count_by_sector)


# Bar plot of total breaches per sector
plt.figure(figsize=(10, 6))
sns.barplot(data=breach_count_by_sector, x='sector', y='breach_count', color='steelblue')
plt.title("Number of Breaches per Sector")
plt.xlabel("Sector")
plt.ylabel("Number of Breaches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("breach_count_by_sector.pdf")
plt.show()


# Convert 'records lost' to numeric (remove commas if present), set errors='coerce' to handle non-numeric values
df['records_lost_numeric'] = pd.to_numeric(df['records lost'].astype(str).str.replace(',', ''), errors='coerce')

# Sum of records lost per sector
record_lost_by_sector = df.groupby('sector')['records_lost_numeric'].sum().reset_index().sort_values(by='records_lost_numeric', ascending=False)
print(record_lost_by_sector)

# Bar plot of total records lost per sector
plt.figure(figsize=(10, 6))
sns.barplot(data=record_lost_by_sector, x='sector', y='records_lost_numeric', color='tomato')
plt.title("Total Records Lost per Sector")
plt.xlabel("Sector")
plt.ylabel("Total Records Lost")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("record_lost_by_sector.pdf")
plt.show()

# Boxplot: Records Lost per Breach by Sector (log scale, mean overlay)
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='sector',
    y='records_lost_numeric',
    data=df,
    color='skyblue',
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
)
plt.yscale('log')  # Set y-axis to log scale
plt.title("Records Lost per Breach by Sector (log scale)")
plt.xlabel("Sector")
plt.ylabel("Records Lost per Breach (log scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("records_lost_per_breach_boxplot_log.pdf")
plt.show()# Calculate mean and standard error for records lost per sector
mean_records_by_sector = df.groupby('sector')['records_lost_numeric'].agg(['mean', 'count', 'std'])
mean_records_by_sector['se'] = mean_records_by_sector['std'] / np.sqrt(mean_records_by_sector['count'])
print(mean_records_by_sector)

# Bar plot of mean records lost per sector with error bars and data points
plt.figure(figsize=(10, 6))
sns.barplot(
    x=mean_records_by_sector.index,
    y=mean_records_by_sector['mean'],
    yerr=mean_records_by_sector['std'],
    color='orange',
    capsize=0.2
)

# Add custom error bars with horizontal heads ("-")
for i, (mean, std) in enumerate(zip(mean_records_by_sector['mean'], mean_records_by_sector['std'])):
    plt.errorbar(
        i, mean, yerr=std, fmt='none', ecolor='black', elinewidth=1.5,
        capsize=8, capthick=2, lolims=False, uplims=False, zorder=3
    )

plt.title("Mean Records Lost per Breach by Sector")
plt.xlabel("Sector")
plt.ylabel("Mean Records Lost")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("mean_records_lost_bar.pdf")
plt.show()

# Pairwise Mann-Whitney U test with Benjamini-Hochberg correction
sector_types = df['sector'].unique()
pairs = list(combinations(sector_types, 2))
pvals = []
for a, b in pairs:
    group_a = df.loc[df['sector'] == a, 'records_lost_numeric'].dropna()
    group_b = df.loc[df['sector'] == b, 'records_lost_numeric'].dropna()
    if len(group_a) > 0 and len(group_b) > 0:
        stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
        pvals.append(p)
    else:
        pvals.append(np.nan)

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

print("\nPairwise Mann-Whitney U test results (Benjamini-Hochberg corrected):")
for idx, (a, b) in enumerate(pairs):
    print(f"{a} vs {b}: p = {pvals_corrected[idx]:.4g} {'*' if reject[idx] else ''}")

