"""
Iris Flower Classification
--------------------------
Classify iris flowers (setosa, versicolor, virginica) from four measurements:
sepal length, sepal width, petal length, petal width.

Steps:
  1. Load the dataset from Scikit-learn
  2. Explore the data
  3. Split into training and test sets
  4. Train several classifiers
  5. Evaluate on the unseen test set (accuracy, precision, recall, F1, confusion matrix)
  6. Cross-validate for a more reliable estimate
  7. Predict the species of new flowers
"""

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
feature_names = list(X.columns)
class_names = [str(name) for name in iris.target_names]

df = X.copy()
df["species"] = y.map(dict(enumerate(class_names)))

# ----------------------------------------------------------------------------
# 2. Explore data
# ----------------------------------------------------------------------------
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Features: {feature_names}")
print(f"Classes : {class_names}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(X.describe().round(2))
print("\nClass distribution:")
print(df["species"].value_counts())
print(f"\nMissing values: {int(X.isna().sum().sum())}")

# Pairplot: shows how well the species separate visually
pair = sns.pairplot(df, hue="species", corner=True, palette="Set2")
pair.figure.suptitle("Iris feature relationships by species", y=1.02)
pair.savefig("pairplot.png", dpi=120, bbox_inches="tight")
plt.close("all")

# ----------------------------------------------------------------------------
# 3. Train / test split
# ----------------------------------------------------------------------------
# stratify=y keeps the same class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\n" + "=" * 70)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ----------------------------------------------------------------------------
# 4. Define models
# ----------------------------------------------------------------------------
# Scale-sensitive models (KNN, SVM, Logistic Regression) get a StandardScaler
# inside a pipeline, so scaling is learned from training data only.
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    ),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "SVM (RBF kernel)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

# ----------------------------------------------------------------------------
# 5. Train and evaluate
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model

    test_acc = accuracy_score(y_test, model.predict(X_test))
    # 5-fold cross-validation on the full dataset
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results.append(
        {
            "Model": name,
            "Test accuracy": round(test_acc, 4),
            "CV mean accuracy": round(cv_scores.mean(), 4),
            "CV std": round(cv_scores.std(), 4),
        }
    )

results_df = pd.DataFrame(results).sort_values(
    ["CV mean accuracy", "Test accuracy"], ascending=False
).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# Bar chart of comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
x_pos = np.arange(len(results_df))
width = 0.38
ax.bar(x_pos - width / 2, results_df["Test accuracy"], width, label="Test accuracy")
ax.bar(x_pos + width / 2, results_df["CV mean accuracy"], width, label="5-fold CV accuracy")
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
ax.set_ylim(0.8, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Model comparison on Iris")
ax.legend()
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Detailed evaluation of the best model
# ----------------------------------------------------------------------------
best_name = results_df.loc[0, "Model"]
best_model = fitted[best_name]
y_pred = best_model.predict(X_test)

print("\n" + "=" * 70)
print(f"DETAILED EVALUATION: {best_name}")
print("=" * 70)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title(f"Confusion matrix: {best_name}")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. Feature importance (Random Forest) - which measurements matter most?
# ----------------------------------------------------------------------------
rf = fitted["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
print("\nRandom Forest feature importances:")
print(importances.sort_values(ascending=False).round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 3.5))
importances.plot.barh(ax=ax, color="seagreen")
ax.set_xlabel("Importance")
ax.set_title("Random Forest feature importance")
fig.tight_layout()
fig.savefig("feature_importance.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8. Predict on new, unseen flowers
# ----------------------------------------------------------------------------
new_flowers = pd.DataFrame(
    [
        [5.0, 3.4, 1.5, 0.2],   # looks like setosa
        [6.0, 2.8, 4.5, 1.3],   # looks like versicolor
        [6.9, 3.1, 5.8, 2.1],   # looks like virginica
    ],
    columns=feature_names,
)
preds = best_model.predict(new_flowers)
print("\n" + "=" * 70)
print(f"PREDICTIONS ON NEW FLOWERS (using {best_name})")
print("=" * 70)
for row, p in zip(new_flowers.values, preds):
    print(f"  measurements {row.tolist()} -> {class_names[p]}")
