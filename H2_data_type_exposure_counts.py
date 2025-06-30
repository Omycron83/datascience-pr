## JUST A TEST FILE
# VISUALIZATION OF HOW MANY TIMES EACH DATA TYPE HAS BEEN LEAKED

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Washington dataset
df1 = pd.read_csv('Washington_DB.csv')
df2 = pd.read_csv('Kaggle_DB_updated.csv')


# ----- PART 1 - WASHINGTON DATASET -----
# Rename InformationType values for clarity
info_type_renames = {
    "Driver's License or Washington ID Card Number": "Driver's License",
    "Email Address and Password/Security Question Answers": "Email & Password",
    "Financial & Banking Information": "Banking Information",
    "Health Insurance Policy or ID Number": "Health Insurance / ID",
    "Unique Private Key (e.g. used to authenticate or sign an electronic record)": "Unique Private Key",
    "Username and Password/Security Question Answers": "Username & Password",
    "Full Date of Birth": "Date of Birth",
    "Social Security Number": "SSN",
}
df1['InformationType'] = df1['InformationType'].replace(info_type_renames)

# Optional: rename industry values too (not used in this plot, but can be useful later)
industry_renames = {
    'Non-Profit/Charity': 'Non-Profit'
}
df1['IndustryType'] = df1['IndustryType'].replace(industry_renames)

# Drop missing values in InformationType column
df1_clean = df1.dropna(subset=['InformationType'])

# Count frequency of each InformationType
info_counts = df1_clean['InformationType'].value_counts().reset_index()
info_counts.columns = ['InformationType', 'Count']

# Keep only the top 10 most frequent information types
top10_info_counts = info_counts.head(10)

# Highlight specific rows within top 10
highlight = ['Name', 'Date of Birth', 'SSN'] # just an example
colors = ['crimson' if info in highlight else 'gray' for info in top10_info_counts['InformationType']]

plt.figure(figsize=(10, 6))
sns.barplot(data=top10_info_counts, y='InformationType', x='Count', palette=colors)

plt.title('Top 10 Most Exposed Information Types', fontsize=20)
plt.xlabel('Number of Breaches')
plt.ylabel('Information Type')
plt.tight_layout()
plt.savefig("Hypothesis2_Plots/Test_H2_Data_type_exposure_counts.png")
plt.show()

# ----- PART 2 - WORDLWIDE DATASET -----

# Clean and map 'data sensitivity' column
df2 = df2.dropna(subset=['data sensitivity'])
sensitivity_map = {
    "1": "Email / Online Info",
    "2": "SSN / Personal",
    "3": "Credit Card",
    "4": "Health / Personal",
    "5": "Full Details"
}
df2['data sensitivity'] = (
    df2['data sensitivity']
    .astype(str)
    .str.extract(r'(\d)')[0]
    .map(sensitivity_map)
)

# Drop rows where mapping failed
df2 = df2.dropna(subset=['data sensitivity'])

# Count frequency of each data sensitivity type
sensitivity_counts = df2['data sensitivity'].value_counts().reset_index()
sensitivity_counts.columns = ['DataType', 'Count']

# Optional: highlight certain types
highlight = ['SSN / Personal', 'Email / Online Info','Credit Card']
colors = ['crimson' if dtype in highlight else 'gray' for dtype in sensitivity_counts['DataType']]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=sensitivity_counts, y='DataType', x='Count', palette=colors)

plt.title('Frequency of Leaked Data Types (Kaggle Dataset)', fontsize=20)
plt.xlabel('Number of Breaches')
plt.ylabel('Data Type')
plt.tight_layout()
plt.savefig("Hypothesis2_Plots/Test_H2_Kaggle_Data_type_exposure_counts.png")
plt.show()
