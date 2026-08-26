import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


DATA_PATH = os.path.join("data", "bank_marketing.csv")
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Basic cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

# Target: yes = customer subscribed/purchased, no = did not subscribe
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop(columns=["y"])
y = df["y"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# One-hot encode categorical variables.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

# A controlled tree helps reduce overfitting and makes the model easier to explain.
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Decision Tree Classifier - Bank Marketing")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1-score         : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Subscription", "Subscription"],
    zero_division=0
))

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["No", "Yes"],
    cmap="Blues",
    ax=ax
)
ax.set_title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# Decision tree visualization
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
tree_model = pipeline.named_steps["classifier"]

plt.figure(figsize=(24, 14))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)
plt.title("Bank Marketing Decision Tree (Top Levels)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree.png"), dpi=200)
plt.close()

# Permutation importance on the original input columns.
result = permutation_importance(
    pipeline, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1"
)

importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances (Permutation Importance)")
plt.xlabel("Mean decrease in F1-score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
plt.close()

importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), header=["importance"])

print("\nVisualizations saved in:", OUTPUT_DIR)
