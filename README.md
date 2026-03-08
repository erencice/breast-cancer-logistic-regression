# 🎗️ Breast Cancer Diagnosis with Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple.svg)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Binary classification of **malignant vs. benign breast tumors** using Logistic Regression on the Wisconsin Breast Cancer Dataset. The pipeline covers a full end-to-end ML workflow — from EDA and feature engineering to class balancing with SMOTE, cross-validation, model interpretability via odds ratios and marginal effects, and rigorous evaluation with overfitting analysis.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | Wisconsin Breast Cancer Dataset (WBCD) |
| **Target** | `diagnosis` — Malignant (1) / Benign (0) |
| **Algorithm** | Logistic Regression |
| **Class Balancing** | SMOTE (Synthetic Minority Oversampling Technique) |
| **Validation** | 5-Fold Cross-Validation (SMOTE inside pipeline) |
| **Interpretability** | Coefficients (β), Odds Ratios, Marginal Effects (MEM) |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

---

## 🗂️ Project Structure

```
breast-cancer-logistic-regression/
│
├── logit.py                  # Main pipeline script
├── breast_cancer.csv         # Dataset
├── requirements.txt          # Dependencies
└── README.md
```

---

## ⚙️ Pipeline Steps

1. **Data Loading & Preprocessing** — Drop `id`, encode `diagnosis` (B→0, M→1)
2. **EDA** — Descriptive statistics, missing value check, target distribution
3. **Feature Engineering** — StandardScaler normalization `(x − μ) / σ`
4. **Train/Test Split** — 80/20 split with `random_state=42`
5. **Cross-Validation** — 5-Fold CV via `imblearn.pipeline` (SMOTE applied per fold, no data leakage)
6. **Class Balancing** — SMOTE applied on full training set for final model training
7. **Model Training** — Logistic Regression (`max_iter=1000`)
8. **Interpretability** — Coefficients (β), Odds Ratios `exp(β)`, Marginal Effects at Mean (MEM)
9. **Evaluation** — Confusion Matrix, Precision-Recall Curve, ROC Curve
10. **Overfitting Check** — Train vs. Test metric comparison

---

## 📊 Model Results

### Cross-Validation Performance (5-Fold)

| | Score |
|---|---|
| **CV Accuracy Scores** | Per-fold results printed at runtime |
| **Mean CV Accuracy** | ~0.98+ |
| **Standard Deviation** | Low variance — stable model |

> SMOTE is wrapped inside an `imblearn.Pipeline` so that synthetic samples are generated **only on each fold's training portion**. The validation fold always contains real observations only, preventing data leakage.

### Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | **0.9825** |
| Precision | **0.9767** |
| Recall | **0.9767** |
| F1-Score | **0.9767** |
| ROC-AUC | **0.9971** |

### Train vs. Test Comparison (Overfitting Check)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.9895 | 0.9825 |
| Precision | 0.9930 | 0.9767 |
| Recall | 0.9860 | 0.9767 |
| F1-Score | 0.9895 | 0.9767 |
| ROC-AUC | 0.9984 | 0.9974 |

> Train and test scores are closely aligned — no overfitting detected.

### Confusion Matrix

|  | Predicted: Benign | Predicted: Malignant |
|--|---|---|
| **Actual: Benign** | 70 | 1 |
| **Actual: Malignant** | 1 | 42 |

### SMOTE — Class Balancing

| | Class 0 (Benign) | Class 1 (Malignant) |
|--|---|---|
| Before SMOTE | 286 | 169 |
| After SMOTE | 286 | 286 |

---

## 🔍 Model Interpretability

Beyond standard metrics, this project goes deeper into understanding **why** the model makes its predictions.

### Top Features by Marginal Effect at Mean (MEM %)

| Feature | MEM % | Direction |
|---------|-------|-----------|
| texture_worst | 42.22% | ↑ |
| radius_se | 37.53% | ↑ |
| symmetry_worst | 35.43% | ↑ |
| concave points_mean | 30.32% | ↑ |
| area_worst | 25.40% | ↑ |
| compactness_se | -18.74% | ↓ |
| compactness_mean | -17.27% | ↓ |

### Top Odds Ratios

| Feature | Odds Ratio |
|---------|-----------|
| texture_worst | 5.41 |
| radius_se | 4.49 |
| symmetry_worst | 4.12 |
| concave points_mean | 3.36 |
| concavity_worst | 3.03 |

- **β Coefficients** — Direction and magnitude of each feature's log-odds effect
- **Odds Ratios** `exp(β)` — Multiplicative change in odds per one-unit increase in a feature
- **Marginal Effects at Mean (MEM)** — Average partial effect of each feature on the predicted probability of malignancy

---

## 💡 Key Findings

- The model correctly classifies **98.25%** of all samples with only **2 misclassifications** out of 114 test observations.
- **5-Fold Cross-Validation** (via `imblearn.Pipeline`) confirms stable generalization — SMOTE is applied per fold to prevent data leakage, ensuring unbiased CV scores.
- `texture_worst`, `radius_se`, and `symmetry_worst` are the strongest predictors of malignancy.
- `compactness_mean` and `compactness_se` show **negative** marginal effects — higher compactness slightly reduces predicted malignancy probability in this model.
- **No overfitting** observed: train and test metrics differ by less than 1% across all metrics.
- Applying **StandardScaler** was critical — without normalization, some marginal effects showed misleading signs.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/erencice/breast-cancer-logistic-regression.git
cd breast-cancer-logistic-regression
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python logit.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

## 📁 Dataset

The **Wisconsin Breast Cancer Dataset** contains 569 samples with 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Classes:** Malignant (212) / Benign (357)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Eren Cice**  
[![GitHub](https://img.shields.io/badge/GitHub-erencice-black?logo=github)](https://github.com/erencice)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-erencice-blue?logo=linkedin)](https://linkedin.com/in/erencice)
