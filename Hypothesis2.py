import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import math

# Create directory for output plots if it doesn't exist
os.makedirs("Hypothesis2_Plots", exist_ok=True)

# Load data
df = pd.read_csv("Washington_DB.csv")

# Rename dictionaries (used only for display)
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
industry_renames = {
    'Non-Profit/Charity': 'Non-Profit'
}

# Clean data
df_clean = df.dropna(subset=['IndustryType', 'InformationType', 'WashingtoniansAffected'])

# Group data
grouped = df_clean.groupby(['IndustryType', 'InformationType'])['WashingtoniansAffected'].sum().reset_index()
industries_sorted = sorted(grouped['IndustryType'].unique())

# Use a colorblind-safe palette
colorblind_palette = [
    "#4477AA",  # blue
    "#EE6677",  # reddish pink
    "#228833",  # green (careful for red-green CB)
    "#CCBB44",  # mustard yellow
    "#66CCEE",  # sky blue
    "#AA3377",  # purple
    "#BBBBBB"   # gray
]

bar_colors = {industry: colorblind_palette[i % len(colorblind_palette)] for i, industry in enumerate(industries_sorted)}

# --- Part 1: Per-industry Top 5 ---
num_industries = len(industries_sorted)
cols = 2
rows = math.ceil(num_industries / cols)


for industry in industries_sorted:
    industry_display = industry_renames.get(industry, industry)
    industry_data = grouped[grouped['IndustryType'] == industry]
    top5 = industry_data.sort_values(by='WashingtoniansAffected', ascending=False).head(5)
    labels = [info_type_renames.get(info, info) for info in top5['InformationType']]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, top5['WashingtoniansAffected'], color=bar_colors[industry])
    plt.title(f"{industry_display}", fontsize=16)
    plt.xlabel("Data Type", fontsize=12)
    plt.ylabel("Affected", fontsize=12)
    plt.ylim(0, top5['WashingtoniansAffected'].max() * 1.15)
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            digits = int(math.floor(math.log10(height))) + 1
            factor = 10 ** (digits - 3)
            rounded = int(round(height / factor)) * factor
            formatted = f"{rounded:,}".replace(",", " ")
            plt.annotate(formatted,
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, -6),
                         textcoords='offset points',
                         ha='center', va='top',
                         fontsize=9, color='white')

    plt.tight_layout()
    # Save each plot separately
    plt.savefig(f"Hypothesis2_Plots/{industry_display.replace(' ', '_')}_Top5_Readable.png", bbox_inches='tight')
    plt.show()


# --- Part 2: Standardized Top 10 by Industry ---
# Total per data type
total_by_info = df_clean.groupby('InformationType')['WashingtoniansAffected'].sum().sort_values(ascending=False)
top10_info_types = total_by_info.head(10).index.tolist()
filtered = df_clean[df_clean['InformationType'].isin(top10_info_types)]

# Group and pivot
grouped_top10 = filtered.groupby(['InformationType', 'IndustryType'])['WashingtoniansAffected'].sum().unstack(fill_value=0)
grouped_percent = grouped_top10.div(grouped_top10.sum(axis=1), axis=0) * 100
grouped_percent = grouped_percent.loc[top10_info_types]

# Rename x-axis labels
new_labels = [info_type_renames.get(info, info) for info in grouped_percent.index]
grouped_percent.index = new_labels

# Sort columns to match order of colors
grouped_percent = grouped_percent[industries_sorted]

# Get color list in correct order
color_list = [bar_colors[ind] for ind in industries_sorted]

# Plot
grouped_percent.plot(kind='bar', stacked=True, figsize=(12, 8), color=color_list)

plt.title("Top 10 Leaked Data Types (Standardized %)", fontsize=16)
plt.xlabel("Data Type Leaked")
plt.ylabel("Percentage by Industry")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Industry Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Hypothesis2_Plots/Top10_DataTypes_ByIndustry_Standardized.png")
plt.show()
