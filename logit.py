# ====================================================
# 1. Importing Libraries
# ====================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from collections import Counter


# ====================================================
# 2. Loading and Copying the Dataset
# ====================================================
file_path = "breast_cancer.csv"
df = pd.read_csv(file_path).copy()


# ====================================================
# 3. Initial Data Inspection
# ====================================================
print("► First 5 Rows:\n", df.head())
print("\n► Data Info:\n")
df.info()


# ====================================================
# 4. Dropping the ID Column & Encoding the Target
# ====================================================
df.drop(["id"], axis=1, inplace=True)

# Encoding the diagnosis variable as numeric (Benign=0, Malignant=1)
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})


# ====================================================
# 5. Descriptive Statistics & Missing Value Check
# ====================================================
print("\n► Descriptive Statistics:\n", df.describe())
print("\n► Missing Value Counts:\n", df.isnull().sum())


# ====================================================
# 6. Target Variable Distribution
# ====================================================
plt.figure(figsize=(6, 6))
sns.countplot(x=df["diagnosis"])
plt.xticks([0, 1], labels=["Benign (0)", "Malignant (1)"])
plt.title("Target Variable Distribution")
plt.show()


# ====================================================
# 7. Feature and Target Separation
# ====================================================
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# ====================================================
# 8. Feature Standardization
# ====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ====================================================
# 9. Train & Test Split
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ====================================================
# 10. Class Balancing with SMOTE
# ====================================================
print("\n► Class Distribution Before SMOTE:", Counter(y_train))
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

print("► Class Distribution After SMOTE:", Counter(y_train))

# ====================================================
# 10.1. Cross-Validation for Model Performance
# ====================================================
logit_cv = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(logit_cv, X_train, y_train, cv=5, scoring="accuracy")

print("\n► Cross-Validation Results (5-Fold):")
print(f"Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# ====================================================
# 11. Building the Logistic Regression Model
# ====================================================
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)


# ====================================================
# 12. Displaying Model Coefficients
# ====================================================
feature_names = df.drop("diagnosis", axis=1).columns
beta = pd.Series(logit.coef_[0], index=feature_names)
print("\n► Coefficients (β):\n", beta)


# ====================================================
# 13. Odds Ratios (Exp(β)) Calculation
# ====================================================
odds_ratios = np.exp(beta)
print("\n► Odds Ratios:\n", odds_ratios)


# ====================================================
# 14. Marginal Effects (MEM) | Average Marginal Effect
# ====================================================
p_mean = logit.predict_proba(X_train)[:, 1].mean()
mem = p_mean * (1 - p_mean) * beta

mem_df = pd.DataFrame({
    "Coefficient (β)": beta,
    "MEM (∂p/∂x)": mem,
    "MEM %": mem * 100
})

print("\n► Marginal Effects at Mean (MEM):\n", mem_df)


# ====================================================
# 15. Model Performance Metrics
# ====================================================
y_pred = logit.predict(X_test)

print("\n► Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n► Accuracy:", accuracy_score(y_test, y_pred))
print("► Precision:", precision_score(y_test, y_pred))
print("► Recall:", recall_score(y_test, y_pred))
print("► F1-Score:", f1_score(y_test, y_pred))
roc_auc = roc_auc_score(y_test, logit.predict_proba(X_test)[:,1])
print("► ROC-AUC:", roc_auc)


# ====================================================
# 16. Confusion Matrix Visualization
# ====================================================
ConfusionMatrixDisplay.from_estimator(logit, X_test, y_test, cmap="Blues_r")
plt.title("Confusion Matrix")
plt.show()


# ====================================================
# 17. Precision-Recall Curve
# ====================================================
PrecisionRecallDisplay.from_estimator(logit, X_test, y_test)
plt.title("Precision-Recall Curve")
plt.show()


# ====================================================
# 18. ROC Curve
# ====================================================
fpr, tpr, thresholds = roc_curve(y_test, logit.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ====================================================
# 19. Overfitting Check – Train vs Test Metrics
# ====================================================
y_train_pred = logit.predict(X_train)
y_test_pred = logit.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, logit.predict_proba(X_train)[:,1])

print("\n► Train vs Test Metrics Comparison")
print(f"Accuracy: Train={train_acc:.4f} | Test={accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: Train={train_prec:.4f} | Test={precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: Train={train_recall:.4f} | Test={recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: Train={train_f1:.4f} | Test={f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: Train={train_auc:.4f} | Test={roc_auc:.4f}")
