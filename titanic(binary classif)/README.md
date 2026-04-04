# Titanic — Binary Classification EDA

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Library](https://img.shields.io/badge/Libraries-pandas%20%7C%20seaborn%20%7C%20sklearn-teal?style=flat-square)
![Type](https://img.shields.io/badge/ML%20Type-Binary%20Classification-coral?style=flat-square)

---

## What is this?

This notebook walks through a complete **Exploratory Data Analysis (EDA)** of the Titanic dataset — the classic entry point for binary classification in machine learning. The goal is to understand the data deeply *before* fitting any model, following a structured pre-modeling workflow.

The central question: **given a passenger's attributes, can we predict whether they survived?**

---

## Dataset

| Property | Detail |
|---|---|
| Source | `seaborn.load_dataset('titanic')` — no download needed |
| Rows | 891 passengers |
| Label | `survived` — 0 = died, 1 = survived |
| ML type | Binary classification |

**Key columns:**

| Column | Type | Description |
|---|---|---|
| `pclass` | numerical | Passenger class (1st, 2nd, 3rd) |
| `sex` | categorical | Gender |
| `age` | numerical | Age in years (~20% missing) |
| `fare` | numerical | Ticket price |
| `sibsp` | numerical | No. of siblings/spouses aboard |
| `parch` | numerical | No. of parents/children aboard |
| `embarked` | categorical | Port of embarkation (S / C / Q) |

---

## What this notebook covers

**Step 1 — Data loading**
Load the dataset directly via seaborn. Inspect shape and first rows.

**Step 2 — Features & label**
Separate `X` (features) from `y` (the `survived` label). Identify numerical vs. categorical columns. Drop leakage-prone derived columns (`alive`, `who`, `adult_male`).

**Step 3 — Missing values**
Identify that `age` (~20%) and `embarked` (2 rows) are missing. Visualize missingness as a bar chart. Explain imputation strategies.

**Step 4 — Descriptive statistics & class balance**
Run `.describe()` on all numerical columns. Check the survival split (~38% survived, ~62% died) and discuss why mild class imbalance matters.

**Step 5 — Data visualization**
- Histograms + KDE plots for `age`, `fare`, `sibsp`, `parch`
- Correlation heatmap (numerical features)
- Survival rate by `sex` and `pclass` bar charts
- KDE overlay — age distribution for survivors vs. non-survivors

**Step 6 — Train/test split (70/30)**
Perform a stratified 70/30 split using `sklearn`. Verify that the survival ratio is preserved in both sets.

**Step 7 — EDA report**
Structured summary of data quality issues, most predictive features, and preprocessing steps needed before modeling.

---

## Key findings

- **Sex** is the single strongest predictor — female survival rate ~74% vs. male ~19%
- **Passenger class** matters significantly — 1st class survived at ~63%, 3rd class at ~24%
- **Fare** is heavily right-skewed with outliers — log transformation recommended
- Mild class imbalance (~38/62 split) — use `class_weight='balanced'` or stratified sampling

---

## Preprocessing needed before modeling

- Impute `age` with median; `embarked` with mode (`'S'`)
- One-hot encode `sex` and `embarked`
- Drop `deck` (77% missing)
- Optional: log-transform `fare`
- Scale numerical features for distance-based models (SVM, k-NN, Logistic Regression)

---

## How to run

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook and paste cells in order
3. Run all — no file uploads or API keys required

```bash
# All dependencies come pre-installed in Colab
pandas | numpy | matplotlib | seaborn | scikit-learn
```

