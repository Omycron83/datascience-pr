import pandas as pd
df = pd.read_csv("Kaggle_DB.csv")
df = df.drop(index=0)


## GLOBAL
# Replace all exact matches of 'financial' with 'finance' in the 'sector' column
df['sector'] = df['sector'].replace('financial', 'Finance')
df['sector'] = df['sector'].replace('finance', 'Finance')
df['sector'] = df['sector'].replace('academic', 'Education')
df['sector'] = df['sector'].astype(str).apply(lambda x: 'Government' if 'government' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'military' if 'military' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'other' if 'misc' in x else x)
df = df[~df['sector'].astype(str).str.contains('legal')]
df = df[~df['sector'].astype(str).str.contains('NGO')]

# Define sectors to merge into "business"
business_keywords = ['app', 'gaming', 'retail', 'tech', 'telecoms', 'transport', 'web']

# Convert anything containing any business keyword into "business"
df['sector'] = df['sector'].astype(str).apply(
    lambda x: 'Business' if any(keyword in x for keyword in business_keywords) else x
)
df = df[~df['sector'].isin(['military', 'other'])]
df.loc[df['sector'].str.strip().str.lower() == 'health', 'sector'] = 'Health'
df.to_csv("Kaggle_DB_updated.csv", index=False)
#print(df['sector'].sort_values().unique())

## LOCAL
df = pd.read_csv("Washington_DB.csv")
df = df[df['IndustryType'] != 'Non-Profit/Charity']
df.to_csv("Washington_DB.csv", index=False)

