import pandas as pd
df = pd.read_csv("Kaggle_DB.csv")

# Replace all exact matches of 'financial' with 'finance' in the 'sector' column
df['sector'] = df['sector'].replace('financial', 'finance')
df['sector'] = df['sector'].astype(str).apply(lambda x: 'government' if 'government' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'military' if 'military' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'misc' if 'misc' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'tech' if 'tech' in x else x)
df['sector'] = df['sector'].astype(str).apply(lambda x: 'web' if 'web' in x else x)

df.to_csv("Kaggle_DB_updated.csv", index=False)