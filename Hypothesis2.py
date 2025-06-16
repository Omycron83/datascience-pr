import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Washington DB.csv")

# Clean and convert 'WashingtoniansAffected' to int
# Remove commas and convert to int
df['WashingtoniansAffected'] = df['WashingtoniansAffected'].astype(str).str.replace(',', '')
df = df[df['WashingtoniansAffected'].str.isnumeric()]  # keep numeric rows only
df['WashingtoniansAffected'] = df['WashingtoniansAffected'].astype(int)

# Group by IndustryType and InformationType, sum affected counts
grouped = df.groupby(['IndustryType', 'InformationType'])['WashingtoniansAffected'].sum().reset_index()

# Get list of unique industries
industries = grouped['IndustryType'].unique()

# Plot top 5 InformationTypes for each Industry
for industry in industries:
    industry_data = grouped[grouped['IndustryType'] == industry]
    top5 = industry_data.sort_values(by='WashingtoniansAffected', ascending=False).head(5)

    plt.figure(figsize=(10,6))
    plt.bar(top5['InformationType'], top5['WashingtoniansAffected'], color='skyblue')
    plt.title(f"Top 5 Breached Data Types in Industry: {industry}")
    plt.xlabel("Data Type")
    plt.ylabel("Total Washingtonians Affected")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()