import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import os


# Read data
wa_raw = pd.read_csv("WA.csv")
#print(wa_raw)

# Overview of the dataset
#print("Number of rows:", wa_raw.shape[0])
#print("Number of columns:", wa_raw.shape[1])
#print("Column names:\n", wa_raw.columns.tolist())
#print("Data types of each column:\n", wa_raw.dtypes)
print("Summary statistics:\n", wa_raw.describe(include='all'))


# Sum total breaches per IndustryType
total_breaches_by_industry = wa_raw['IndustryType'].value_counts().sort_values(ascending=False)
print("total breaches per IndustryType: \n", total_breaches_by_industry)

# Bar plot of total breaches per IndustryType
plt.figure(figsize=(8, 6))
total_breaches_by_industry.plot(kind='bar', color='steelblue')
plt.title("Total Number of Breaches by Industry Type")
plt.xlabel("Industry Type")
plt.ylabel("Number of Breaches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("breach_bar_plot.pdf")
plt.show()


# Total number of people affected per IndustryType
total_affected_by_industry = wa_raw.groupby('IndustryType')['WashingtoniansAffected'].sum().sort_values(ascending=False)
print("people affected per IndustryType: \n", total_affected_by_industry)

# Bar plot of total people affected per IndustryType
plt.figure(figsize=(8, 6))
total_affected_by_industry.plot(kind='bar', color='crimson')
plt.title("Total Number of People Affected by Industry Type")
plt.xlabel("Industry Type")
plt.ylabel("Number of People Affected")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("total_affected_bar_plot.pdf")
plt.show()


# Boxplot: WashingtoniansAffected by IndustryType
plt.figure(figsize=(10, 6))
sns.boxplot(x='IndustryType', y='WashingtoniansAffected', data=wa_raw, color='skyblue')
plt.title("Washingtonians Affected by Industry Type")
plt.xlabel("Industry Type")
plt.ylabel("Number of Washingtonians Affected")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("waff_boxplot.pdf")
plt.show()


# Boxplot: WashingtoniansAffected by IndustryType (with log scale and mean overlay)
plt.figure(figsize=(10, 6))
sns.boxplot(x='IndustryType', y='WashingtoniansAffected', data=wa_raw, color='skyblue', showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"})
plt.yscale('log')  # Set y-axis to log scale
plt.title("Washingtonians Affected by Industry Type (log scale)")
plt.xlabel("Industry Type")
plt.ylabel("Number of Washingtonians Affected (log scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("waff_boxplot_log.pdf")
plt.show()



# Calculate mean and standard error for WashingtoniansAffected per IndustryType
mean_waff_by_industry = wa_raw.groupby('IndustryType')['WashingtoniansAffected'].agg(['mean', 'count', 'std'])
mean_waff_by_industry['se'] = mean_waff_by_industry['std'] / np.sqrt(mean_waff_by_industry['count'])
print(mean_waff_by_industry)

# Bar plot of mean WashingtoniansAffected by IndustryType with error bars and data points
plt.figure(figsize=(10, 6))
sns.barplot(
    x=mean_waff_by_industry.index, 
    y=mean_waff_by_industry['mean'], 
    yerr=mean_waff_by_industry['std'], 
    color='orange', 
    capsize=0.2
)'

# Add custom error bars with horizontal heads ("-")
for i, (mean, std) in enumerate(zip(mean_waff_by_industry['mean'], mean_waff_by_industry['std'])):
    plt.errorbar(
        i, mean, yerr=std, fmt='none', ecolor='black', elinewidth=1.5,
        capsize=8, capthick=2, lolims=False, uplims=False, zorder=3
    )

plt.title("Mean Washingtonians Affected per Breach by Industry Type")
plt.xlabel("Industry Type")
plt.ylabel("Mean Number of Washingtonians Affected")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("mean_waff_bar.pdf")
plt.show()

# Pairwise Wilcoxon (Mann-Whitney U) test with Benjamini-Hochberg correction
industry_types = wa_raw['IndustryType'].unique()
pairs = list(combinations(industry_types, 2))
pvals = []
for a, b in pairs:
    group_a = wa_raw.loc[wa_raw['IndustryType'] == a, 'WashingtoniansAffected'].dropna()
    group_b = wa_raw.loc[wa_raw['IndustryType'] == b, 'WashingtoniansAffected'].dropna()
    if len(group_a) > 0 and len(group_b) > 0:
        stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
        pvals.append(p)
    else:
        pvals.append(np.nan)

# Benjamini-Hochberg correction
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

print("\nPairwise Mann-Whitney U test results (Benjamini-Hochberg corrected):")
for idx, (a, b) in enumerate(pairs):
    print(f"{a} vs {b}: p = {pvals_corrected[idx]:.4g} {'*' if reject[idx] else ''}")
