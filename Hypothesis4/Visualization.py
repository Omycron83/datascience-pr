import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# How do records lost vary by method?
# -----------------------------------------------------

# Load the dataset
df = pd.read_csv('Kaggle_DB_updated.csv')

# Clean and preprocess
df = df.dropna(subset=['records lost', 'method'])
df['records lost'] = df['records lost'].str.replace(',', '', regex=False).astype(int)
df['method'] = df['method'].str.strip()
df['method'] = df['method'].replace('poor security ', 'poor security')
df['method'] = df['method'].replace('lost device ', 'lost device')

# -----------------------------------------------------
# Separate box-plot for Each Method (using FacetGrid)
# -----------------------------------------------------
# Plot box plot of records lost by method
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x='method',
    y='records lost',
    palette='pastel'
)
plt.yscale('log')  # Log scale to handle skew
plt.xlabel('Breach Method', fontsize=14)
plt.ylabel('Records Lost (log scale)', fontsize=14)
plt.title('Distribution of Records Lost by Breach Method', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig("Hypothesis4/Hypothesis4_Plots/records_lost_by_method_boxplot.svg")
plt.show()

# -----------------------------------------------------
# Number of Breaches by Method (Count Plot)
# -----------------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    y='method',
    order=df['method'].value_counts().index,
    palette='pastel'
)
plt.xlabel('Number of Breaches', fontsize=14)
plt.ylabel('Breach Method', fontsize=14)
plt.title('Number of Breaches by Method', fontsize=16)
plt.tight_layout()
plt.savefig("Hypothesis4/Hypothesis4_Plots/breach_counts_by_method.svg")
plt.show()
