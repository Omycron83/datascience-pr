# ---------------------------------------------------------
# Hypothesis 2: Data Breaches by Industry and Data Type
# This script analyzes data breaches in Washington State and worldwide, focusing on the number of breaches and people affected by industry type.
# It includes visualizations and statistical tests to understand the impact of data breaches across different sectors.
# ---------------------------------------------------------     

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import math

def main():
    # Load data
    df = pd.read_csv("data/Washington_DB.csv")
    kaggle_df = pd.read_csv('data/Kaggle_DB_updated.csv')

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
        plt.savefig(f"Hypothesis2/Hypothesis2_Plots/H2_{industry_display.replace(' ', '_')}_Top5_WA.png", bbox_inches='tight')
        plt.show()


    # --- Part 2: Standardized Top 10 by Industry in Washington (MODIFIED AXIS & COLORS) ---
    # Total per data type
    total_by_info = df_clean.groupby('InformationType')['WashingtoniansAffected'].sum().sort_values(ascending=False)
    top10_info_types = total_by_info.head(10).index.tolist()
    filtered = df_clean[df_clean['InformationType'].isin(top10_info_types)]

    # Group and pivot: now pivot so sector is index
    grouped_top10 = filtered.groupby(['IndustryType', 'InformationType'])['WashingtoniansAffected'].sum().unstack(fill_value=0)
    grouped_percent = grouped_top10.div(grouped_top10.sum(axis=1), axis=0) * 100

    # Restrict to sectors that reported top 10 types
    grouped_percent = grouped_percent.loc[industries_sorted]

    # Rename columns (data types)
    new_columns = [info_type_renames.get(info, info) for info in grouped_percent.columns]
    grouped_percent.columns = new_columns

    # Use consistent color palette for info types
    info_type_list = grouped_percent.columns.tolist()
    info_colors = {info: colorblind_palette[i % len(colorblind_palette)] for i, info in enumerate(info_type_list)}
    color_list = [info_colors[info] for info in info_type_list]

    # Plot
    grouped_percent.plot(kind='bar', stacked=True, figsize=(12, 8), color=color_list)

    plt.title("Standardized % of Top 10 Data Types by Industry (Washington)", fontsize=16)
    plt.xlabel("Industry Type")
    plt.ylabel("Percentage by Data Type")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Hypothesis2/Hypothesis2_Plots/H2_Standardized_Washington_Modified.png")
    plt.show()


    ## ------ PART 3 & 4 DATASET ------

    # Load second dataset (skip first row if it’s notes)
    df2_raw = pd.read_csv("data/Kaggle_DB_updated.csv")

    # Drop rows with missing necessary fields
    df2_clean = df2_raw.dropna(subset=['sector', 'data sensitivity', 'records lost']).copy()

    # Clean and convert 'records lost' to numeric
    df2_clean.loc[:, 'records lost'] = (
        df2_clean['records lost']
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r'(\d+)')[0]
    )
    df2_clean.loc[:, 'records lost'] = pd.to_numeric(df2_clean['records lost'], errors='coerce')
    df2_clean = df2_clean.dropna(subset=['records lost'])

    # Map sensitivity to readable names
    sensitivity_map = {
        "1": "Email / Online Info (1.0)",
        "2": "SSN / Personal (2.0)",
        "3": "Credit Card (3.0)",
        "4": "Health / Personal (4.0)",
        "5": "Full Details (5.0)"
    }

    mapped = (
        df2_clean['data sensitivity']
        .astype(str)
        .str.extract(r'(\d)')[0]
        .map(sensitivity_map)
    )

    # Explicitly cast to object dtype before assigning back
    df2_clean.loc[:, 'data sensitivity'] = mapped.astype(object)


    # Drop rows where mapping failed
    df2_clean = df2_clean.dropna(subset=['data sensitivity'])

    # Color palette (Paul Tol safe palette)
    tol_colors = [
        "#4477AA",  # blue
        "#EE6677",  # reddish pink
        "#228833",  # green (careful for red-green CB)
        "#CCBB44",  # mustard yellow
        "#66CCEE",  # sky blue
        "#AA3377",  # purple
        "#BBBBBB"   # gray
    ]

    sectors = sorted(df2_clean['sector'].unique())
    sector_colors = {sector: tol_colors[i % len(tol_colors)] for i, sector in enumerate(sectors)}

    # Group data by sector and data sensitivity
    grouped = df2_clean.groupby(['sector', 'data sensitivity'])['records lost'].sum().reset_index()

    # --- PART 4: Standardized % by Sector (MODIFIED AXIS & COLORS) ---

    pivot_table = grouped.pivot(index='sector', columns='data sensitivity', values='records lost').fillna(0)
    pivot_table = pivot_table.infer_objects()
    pivot_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Ensure consistent column order based on sensitivity_map values
    info_order = list(sensitivity_map.values())
    pivot_percent = pivot_percent[info_order]

    # Color palette by data type
    info_colors = {info: tol_colors[i % len(tol_colors)] for i, info in enumerate(info_order)}
    color_list = [info_colors[info] for info in info_order]

    # Plot
    pivot_percent.plot(kind='bar', stacked=True, figsize=(12, 8), color=color_list)

    plt.title("Leaked Data Types by Sector (Standardized %, Worldwide)", fontsize=16)
    plt.xlabel("Sector")
    plt.ylabel("% by Data Type")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Hypothesis2/Hypothesis2_Plots/H2_Standardized_Worldwide_Modified.png")
    plt.show()

if __name__ == "__main__":
    main()