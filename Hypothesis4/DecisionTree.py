import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

print(f"Dataset shape: {df.shape}")
print(f"Unique methods: {df['method'].unique()}")
print(f"Unique sectors: {df['sector'].unique()}")
print(f"Unique data sensitivities: {df['data sensitivity'].unique()}")

# One-hot encode categorical variables (no ordering assumed)
sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
le_method = LabelEncoder()  # Keep this for target variable

# Prepare data
X = pd.concat([
    df[['records lost', 'data sensitivity']].rename(columns={'records lost': 'records_lost'}),
    sector_dummies
], axis=1)

y = le_method.fit_transform(df['method'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize base model
dt = DecisionTreeClassifier(random_state=42)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
dt = grid_search.best_estimator_

# Predictions
y_pred = dt.predict(X_test)

# Evaluation
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_method.classes_))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_method.classes_, yticklabels=le_method.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("./Hypothesis4/Hypothesis4_Plots/confusion_matrix.svg")
plt.show()

# Feature importance
plt.figure(figsize=(12, 6))
feature_names = X.columns.tolist()
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig("./Hypothesis4/Hypothesis4_Plots/feature_importance.svg")
plt.show()

# Visualize decision tree
plt.figure(figsize=(25, 15))
plot_tree(dt, 
          feature_names=feature_names,
          class_names=le_method.classes_,
          filled=True,
          rounded=True,
          fontsize=10,
          impurity=False,    # Remove gini values
)      # Remove sample counts
plt.title('Decision Tree for Method Prediction', fontsize=16)
plt.tight_layout()
plt.savefig("./Hypothesis4/Hypothesis4_Plots/decision_tree.svg")
plt.show()

# Sample predictions on test set
print("\nSample Predictions:")
for i in range(min(10, len(X_test))):
    pred_class = le_method.classes_[y_pred[i]]
    true_class = le_method.classes_[y_test[i]]
    print(f"Predicted: {pred_class}, Actual: {true_class}")

# Additional analysis - show data distribution by features
print("\nData distribution by sector:")
print(pd.crosstab(df['sector'], df['method']))

print("\nData distribution by data sensitivity:")
print(pd.crosstab(df['data sensitivity'], df['method']))

print("\nRecords lost statistics by method:")
print(df.groupby('method')['records lost'].describe())
