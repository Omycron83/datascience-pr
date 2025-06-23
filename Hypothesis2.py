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
kaggle_df = pd.read_csv('Kaggle_DB_updated.csv')

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

# --- Part 1: Per-industry Top 5 in Washington---
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
    plt.title(f"{industry_display} in Washington", fontsize=16)
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
    # plt.show()


# --- Part 2: Standardized Top 10 by Industry in Washington ---
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

plt.title("Top 10 Leaked Data Types (Standardized %) in Washington", fontsize=16)
plt.xlabel("Data Type Leaked")
plt.ylabel("Percentage by Industry")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Industry Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Hypothesis2_Plots/Top10_DataTypes_ByIndustry_Standardized.png")
# plt.show()



# -------------------------
# --- Part 3 & 4: New Dataset ---
# -------------------------

# Load second dataset
df2_raw = pd.read_csv("Kaggle_DB_updated.csv", skiprows=1)

print(df2_raw.columns)
# Rename relevant columns
df2_raw = df2_raw.rename(columns={
    'records lost': 'records_lost',
    'sector': 'industry',
    'data sensitivity': 'information_type'
})

# Drop rows with missing sector or information_type
df2_clean = df2_raw.dropna(subset=['industry', 'information_type', 'records_lost'])

# Convert records_lost to numeric
df2_clean['records_lost'] = df2_clean['records_lost'].astype(str).str.replace(",", "").str.extract(r'(\d+)')[0]
df2_clean['records_lost'] = pd.to_numeric(df2_clean['records_lost'], errors='coerce')
df2_clean = df2_clean.dropna(subset=['records_lost'])

# Map sensitivity for readability
sensitivity_renames = {
    "1": "Email / Online Info",
    "2": "SSN / Personal",
    "3": "Credit Card",
    "4": "Health / Personal",
    "5": "Full Details"
}
df2_clean['information_type'] = df2_clean['information_type'].astype(str).str.extract(r'(\d)')[0]
df2_clean['information_type'] = df2_clean['information_type'].map(sensitivity_renames)

# Set up colors using Paul Tol's green-red safe palette
tol_colors = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100"
]
sectors_sorted = sorted(df2_clean['industry'].dropna().unique())
sector_colors = {sector: tol_colors[i % len(tol_colors)] for i, sector in enumerate(sectors_sorted)}

# --- PART 3: Bar Chart per Sector ---
grouped = df2_clean.groupby(['industry', 'information_type'])['records_lost'].sum().reset_index()

cols = 2
rows = math.ceil(len(sectors_sorted) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
axes = axes.flatten()

for i, sector in enumerate(sectors_sorted):
    sector_data = grouped[grouped['industry'] == sector].sort_values('records_lost', ascending=False)
    labels = sector_data['information_type']
    values = sector_data['records_lost']

    ax = axes[i]
    bars = ax.bar(labels, values, color=sector_colors[sector])
    ax.set_title(f"{sector}", fontsize=14)
    ax.set_xlabel("Data Type", fontsize=11)
    ax.set_ylabel("Records Lost", fontsize=11)
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            digits = int(math.floor(math.log10(height))) + 1
            factor = 10 ** (digits - 3)
            rounded = int(round(height / factor)) * factor
            formatted = f"{rounded:,}".replace(",", " ")
            ax.annotate(formatted,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -6),
                        textcoords='offset points',
                        ha='center', va='top',
                        fontsize=8, color='white')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3, w_pad=3)
plt.suptitle("Records Lost by Data Type and Sector (New Dataset)", fontsize=18)
plt.savefig("Hypothesis2_Plots/Part3_PerSector.png", bbox_inches='tight')
plt.show()

# --- PART 4: Standardized Stacked Bar Chart ---

grouped_pct = grouped.pivot(index='information_type', columns='industry', values='records_lost').fillna(0)
grouped_pct = grouped_pct.div(grouped_pct.sum(axis=1), axis=0) * 100

grouped_pct = grouped_pct.loc[sensitivity_renames.values()]
color_list = [sector_colors[sec] for sec in grouped_pct.columns]

grouped_pct.plot(kind='bar', stacked=True, figsize=(12, 8), color=color_list)

plt.title("Standardized % of Data Types by Sector (New Dataset)", fontsize=16)
plt.xlabel("Data Type")
plt.ylabel("Percentage by Sector")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Hypothesis2_Plots/Part4_Standardized.png")
plt.show()
