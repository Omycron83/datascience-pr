import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('Kaggle_DB_updated.csv')  # Replace with your actual file path

# Basic data cleaning
df = df.dropna(subset=['records lost', 'sector', 'data sensitivity', 'method'])
df['records lost'] = df['records lost'].str.replace(',', '', regex=False).astype(int)
df['data sensitivity'] = pd.to_numeric(df['data sensitivity'], errors='coerce')
df = df.dropna(subset=['records lost', 'data sensitivity'])

# Applying log-encoding to reduce skew
df['records lost'] = np.log1p(df['records lost'])  # reduce skew

# Clean up method categories (remove extra spaces and duplicates)
df['method'] = df['method'].str.strip()
df['method'] = df['method'].replace('poor security ', 'poor security')
df['method'] = df['method'].replace('lost device ', 'lost device')
df.drop(df[df['method'] == 'hacked'].index, inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Unique methods: {df['method'].unique()}")
print(f"Unique sectors: {df['sector'].unique()}")
print(f"Unique data sensitivities: {df['data sensitivity'].unique()}")

# One-hot encode categorical variables
sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
le_method = LabelEncoder()

# Prepare features and target
X = pd.concat([
    df[['records lost', 'data sensitivity']].rename(columns={'records lost': 'records_lost'}),
    sector_dummies
], axis=1)
y = le_method.fit_transform(df['method'])

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': list(range(2, 11)),
    'min_samples_split': list(range(2, 11)),
    'min_samples_leaf': list(range(1, 5)),
    'criterion': ['gini', 'entropy'],
}

# Initialize base model
dt = DecisionTreeClassifier(random_state=42)

# Grid search with 10-fold cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Evaluate best estimator with 10-fold CV
best_dt = grid_search.best_estimator_
cv_scores = cross_val_score(best_dt, X, y, cv=5, scoring='accuracy')
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Cross-validated Accuracy (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit final model on full dataset
best_dt.fit(X, y)

# Visualization and analysis as before
plt.figure(figsize=(25, 15))
plot_tree(best_dt, 
          feature_names=X.columns.tolist(),
          class_names=le_method.classes_,
          filled=True,
          rounded=True,
          fontsize=10,
          impurity=False)
plt.title('Final Decision Tree Trained on All Data')
plt.tight_layout()
plt.savefig("./Hypothesis4/Hypothesis4_Plots/final_decision_tree.svg")
plt.show()

# Feature importance
plt.figure(figsize=(12, 6))
importances = best_dt.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=45)
plt.title('Feature Importance (Final Model)')
plt.tight_layout()
plt.savefig("./Hypothesis4/Hypothesis4_Plots/final_feature_importance.svg")
plt.show()