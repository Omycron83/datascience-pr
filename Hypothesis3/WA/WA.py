import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, f_oneway, ttest_ind  
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import os


# Read data
df = pd.read_csv("Washington_DB.csv")
#print(wa_raw)

# Overview of the dataset
#print("Number of rows:", wa_raw.shape[0])
#print("Number of columns:", wa_raw.shape[1])
#print("Column names:\n", wa_raw.columns.tolist())
#print("Data types of each column:\n", wa_raw.dtypes)
print("Summary statistics:\n", df.describe(include='all'))


# Sum total breaches per IndustryType
total_breaches_by_industry = df['IndustryType'].value_counts().sort_values(ascending=False)
print("total breaches per IndustryType: \n", total_breaches_by_industry)

# Bar plot of total breaches per IndustryType
plt.figure(figsize=(8, 6))
total_breaches_by_industry.plot(kind='bar', color='steelblue')
plt.title("Total Number of Breaches per Sector - Local Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of Breaches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("breach_bar_plot.png")
plt.show()


# Total number of people affected per IndustryType
total_affected_by_industry = df.groupby('IndustryType')['WashingtoniansAffected'].sum().sort_values(ascending=False)
print("people affected per IndustryType: \n", total_affected_by_industry)

# Bar plot of total people affected per IndustryType
plt.figure(figsize=(8, 6))
total_affected_by_industry.plot(kind='bar', color='crimson')
plt.title("Total Number of People Affected per Sector - Local Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of People Affected")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("total_affected_bar_plot.png")
plt.show()


# Boxplot: WashingtoniansAffected by IndustryType (with log scale and mean overlay)
order = df.groupby('IndustryType')['WashingtoniansAffected'].mean().sort_values(ascending=False).index

plt.figure(figsize=(8, 6))
sns.boxplot(
    x='IndustryType',
    y='WashingtoniansAffected',
    data=df,
    color='skyblue',
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
    order=order  # <-- Sort x-axis by mean
)
plt.yscale('log')  # Set y-axis to log scale
plt.title("People Affected per Breach by Sector - Local Dataset")
plt.xlabel("Sector")
plt.ylabel("Number of People Affected per Breach (log scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("waff_boxplot_log.png")
plt.show()


# One-way ANOVA test to check for overall significance
groups = [df.loc[df['IndustryType'] == ind, 'WashingtoniansAffected'].dropna() for ind in df['IndustryType'].unique()]
f_stat, p_anova = f_oneway(*groups)
print(f"\nOne-way ANOVA: F = {f_stat:.4f}, p = {p_anova:.4g}")
if p_anova < 0.05:
    print("Result: Significant differences exist between at least some IndustryTypes.")
else:
    print("Result: No significant difference between IndustryTypes.")


# Pairwise t-tests with Bonferroni correction
industry_types = df['IndustryType'].unique()
pairs = list(combinations(industry_types, 2))
ttest_pvals = []

for a, b in pairs:
    group_a = df.loc[df['IndustryType'] == a, 'WashingtoniansAffected'].dropna()
    group_b = df.loc[df['IndustryType'] == b, 'WashingtoniansAffected'].dropna()
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