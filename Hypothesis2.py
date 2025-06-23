import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

# os.makedirs("Hypothesis2_Plots", exist_ok=True)

# Load data
df = pd.read_csv("Washington DB.csv")

# Define renaming dictionaries
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

# Clean dataset
df_clean = df.dropna(subset=['IndustryType', 'InformationType', 'WashingtoniansAffected'])

# Group by original names
grouped = df_clean.groupby(['IndustryType', 'InformationType'])['WashingtoniansAffected'].sum().reset_index()

# Get list of unique industries
industries = grouped['IndustryType'].unique()

# Plot top 5 InformationTypes per industry
for industry in industries:
    industry_display = industry_renames.get(industry, industry)  # Rename if in dictionary
    industry_data = grouped[grouped['IndustryType'] == industry]
    top5 = industry_data.sort_values(by='WashingtoniansAffected', ascending=False).head(5)

    # Apply label renaming only to x-axis labels
    labels = [info_type_renames.get(info, info) for info in top5['InformationType']]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, top5['WashingtoniansAffected'], color='skyblue')
    plt.title(f"Top 5 Breached Data Types in Industry: {industry_display}")
    plt.xlabel("Data Type")
    plt.ylabel("Total Washingtonians Affected")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #plt.savefig(f"Hypothesis2_Plots/{industry_display}_Top5.png")
    plt.show()
