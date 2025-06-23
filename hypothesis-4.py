import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_data.csv' with your actual file path)
df = pd.read_csv('Kaggle_DB.csv')

# Clean and preprocess
def parse_records(val):
    if pd.isnull(val):
        return None
    val = str(val).replace(',', '').strip().lower()
    if 'm' in val:
        return int(val.replace('m', '')) * 1_000_000
    try:
        return int(val)
    except ValueError:
        return None

df['records lost'] = df['records lost']#.apply(parse_records)
df['data sensitivity'] = pd.to_numeric(df['data sensitivity'], errors='coerce')

# Drop rows with missing required values
df = df.dropna(subset=['records lost', 'data sensitivity'])

# Ensure data sensitivity is integer and limited to 1-5
df = df[df['data sensitivity'].isin([1, 2, 3, 4, 5])]
df['data sensitivity'] = df['data sensitivity'].astype(int)

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='data sensitivity',
    y='records lost',
    data=df,
    palette='pastel'
)
plt.yscale('log')
plt.xlabel('Data Sensitivity (1=Low, 5=High)', fontsize=14)
plt.ylabel('Records Lost', fontsize=14)
plt.title('Distribution of Records Lost by Data Sensitivity Level', fontsize=16)
plt.tight_layout()

plt.savefig("x.svg")