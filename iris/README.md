# Iris — Multiclass Classification EDA

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Library](https://img.shields.io/badge/Libraries-pandas%20%7C%20seaborn%20%7C%20sklearn-teal?style=flat-square)
![Type](https://img.shields.io/badge/ML%20Type-Multiclass%20Classification-purple?style=flat-square)

---

## What is this?

This notebook walks through a complete **Exploratory Data Analysis (EDA)** of the Iris dataset — the go-to example for multiclass classification. Unlike binary classification (yes/no), here the model must choose between **3 possible output classes**.

As a bonus, the notebook also demonstrates how an **unsupervised K-Means** algorithm can re-discover the 3 species without ever being shown the labels — a powerful illustration of the supervised vs. unsupervised distinction.

The central question: **given four flower measurements, which of the 3 iris species does it belong to?**

---

## Dataset

| Property | Detail |
|---|---|
| Source | `sklearn.datasets.load_iris()` — no download needed |
| Rows | 150 flowers |
| Label | `species` — setosa / versicolor / virginica |
| ML type | Multiclass classification (3 classes) |

**Key columns:**

| Column | Type | Description |
|---|---|---|
| `sepal length (cm)` | numerical | Length of the sepal |
| `sepal width (cm)` | numerical | Width of the sepal |
| `petal length (cm)` | numerical | Length of the petal |
| `petal width (cm)` | numerical | Width of the petal |
| `species` | categorical (label) | One of: setosa, versicolor, virginica |

> All 4 features are `float64` — **no categorical encoding required**.

---

## What this notebook covers

**Step 1 — Data loading**
Load via `sklearn.datasets.load_iris()`. Add a human-readable `species` column alongside the numeric `target`.

**Step 2 — Features & label**
Define `X` (4 numerical measurements) and `y` (`species`). Note that unlike Titanic, zero preprocessing of data types is needed.

**Step 3 — Missing values**
Confirm zero missing values. Emphasize that even clean datasets should be checked — in real pipelines, nulls can arrive unexpectedly.

**Step 4 — Descriptive statistics & class balance**
Run `.describe()`. Verify that Iris is **perfectly balanced** — exactly 50 samples per class. Discuss why this is rare in real-world data.

**Step 5 — Data visualization**
- KDE plots for all 4 features, split by species — look for separation
- Correlation heatmap — check for multicollinearity between the 4 measurements
- **Pair plot** — the signature Iris visualization; all feature combinations at once, colored by species
- Box plots — median and spread of each feature per species

**Step 6 — Supervised vs. unsupervised (bonus)**
Run K-Means (k=3) on the raw features with no labels. Plot the resulting clusters side-by-side with the true species labels. The near-identical result confirms that the 3 species form genuinely distinct groups in feature space.

**Step 7 — Train/test split (80/20)**
Use an 80/20 split (rather than 70/30) because Iris only has 150 rows — preserving more training data matters. Verify stratification across all 3 classes.

**Step 8 — EDA report**
Structured summary of data quality, most predictive features, and preprocessing steps needed before modeling.

---

## Key findings

- **Petal length** and **petal width** are the most discriminative features — near-perfect species separation visible in KDE and pair plots
- **Setosa** is completely linearly separable from the other two species on petal measurements
- **Versicolor and Virginica** have slight overlap — models need to handle this boundary carefully
- High correlation (~0.96) between petal length and petal width — consider dropping one for linear models
- K-Means (k=3) re-discovered the true species groupings without ever seeing the labels

---

## Preprocessing needed before modeling

- **Feature scaling** — StandardScaler or MinMaxScaler required for SVM, k-NN, Logistic Regression, and K-Means
- **Label encoding** — encode `species` as 0/1/2 for most sklearn classifiers
- No imputation needed, no categorical encoding needed

---

## How to run

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook and paste cells in order
3. Run all — no file uploads or API keys required

```bash
# All dependencies come pre-installed in Colab
pandas | numpy | matplotlib | seaborn | scikit-learn
```
