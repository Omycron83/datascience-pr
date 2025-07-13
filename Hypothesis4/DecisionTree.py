import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, _tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import re

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
class_labels = le_method.classes_  # list of method names in order of encoding

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

# Sorted by frequency (Seaborn plots use this)
ordered_methods = df['method'].value_counts().index.tolist()

colorblind_palette = [
    "#4477AA",  # blue
    "#EE6677",  # reddish pink
    "#228833",  # green
    "#CCBB44",  # mustard yellow
    "#66CCEE",  # sky blue
    "#AA3377",  # purple
    "#BBBBBB"   # gray
]

method_to_color = {
    method: colorblind_palette[i % len(colorblind_palette)]
    for i, method in enumerate(ordered_methods)
}


# Helper: get predicted class for a node
def predicted_class(value_array):
    return np.argmax(value_array)

# Helper: format float nicely
def fmt(val):
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    elif val >= 1e3:
        return f"{val / 1e3:.0f}K"
    else:
        return f"{val:.0f}"

# Build the Graphviz Digraph
dot = Digraph()
dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica')

tree = best_dt.tree_
feature_names = X.columns

# Encode method labels using this fixed order
method_to_index = {method: i for i, method in enumerate(ordered_methods)}
index_to_method = {i: method for method, i in method_to_index.items()}

y = df['method'].map(method_to_index)
class_labels = ordered_methods  # Same as index_to_method.values()

# Use consistent class color mapping
class_colors = {
    i: method_to_color[method]
    for i, method in index_to_method.items()
}

# Label formatting helper
def fmt(val):
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    elif val >= 1e3:
        return f"{val / 1e3:.0f}K"
    else:
        return f"{val:.0f}"

# Recursive function to draw nodes and edges
def add_nodes_edges(dot, node_id=0):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    value = tree.value[node_id][0]

    # Class probabilities
    sorted_class_indices = np.argsort(value)[::-1]
    top_idx = sorted_class_indices[0]
    second_idx = sorted_class_indices[1] if len(sorted_class_indices) > 1 else None
    third_idx = sorted_class_indices[2] if len(sorted_class_indices) > 2 else None

    top_class = class_labels[top_idx]
    second = (
        f"\n2nd: {class_labels[second_idx]}"
    )
    third = (
        f"\n3rd: {class_labels[third_idx]}"
    )

    color = class_colors[top_idx]

    # Node label logic
    if left == _tree.TREE_LEAF:
        label = f"{top_class} {second} {third}"
    else:
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # Format label differently for one-hot-encoded sector columns
        if feature.startswith('sector_'):
            sector_name = feature.replace('sector_', '')
            label = f"Is the sector {sector_name}?\n"
        elif feature == 'records_lost':
            threshold_val = np.expm1(threshold)
            label = f"Is the {feature} ≤ {fmt(threshold_val)}\n"
        else:
            label = f"Is the {feature} ≤ {fmt(threshold)}\n"

    dot.node(str(node_id), label=label, fillcolor=color)

    # Recurse for child nodes
    if left != _tree.TREE_LEAF:
        dot.edge(str(node_id), str(left), label="True")
        add_nodes_edges(dot, left)
        dot.edge(str(node_id), str(right), label="False")
        add_nodes_edges(dot, right)

# Start recursive tree build
add_nodes_edges(dot)

# Save to file
dot.format = 'svg'
dot.render('./Hypothesis4/Hypothesis4_Plots/final_decision_tree', cleanup=True)

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