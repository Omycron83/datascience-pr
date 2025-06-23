import pandas as pd
df = pd.read_csv("Kaggle_DB.csv")
df = df.drop(index=0)

# Replace all exact matches of 'financial' with 'finance' in the 'sector' column
df['sector'] = df['sector'].replace('financial', 'finance')
df['sector'] = df['sector'].replace('academic', 'education')
df['sector'] = df['sector'].astype(str).apply(lambda x: 'government' if 'government' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'military' if 'military' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'other' if 'misc' in x else x)
df = df[~df['sector'].astype(str).str.contains('legal')]
df = df[~df['sector'].astype(str).str.contains('NGO')]

# Define sectors to merge into "business"
business_keywords = ['app', 'gaming', 'retail', 'tech', 'telecoms', 'transport', 'web']

# Convert anything containing any business keyword into "business"
df['sector'] = df['sector'].astype(str).apply(
    lambda x: 'business' if any(keyword in x for keyword in business_keywords) else x
)

df.to_csv("Kaggle_DB_updated.csv", index=False)