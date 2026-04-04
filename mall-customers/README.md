# Mall Customers — Clustering / Unsupervised EDA

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Library](https://img.shields.io/badge/Libraries-pandas%20%7C%20seaborn%20%7C%20sklearn-teal?style=flat-square)
![Type](https://img.shields.io/badge/ML%20Type-Unsupervised%20Clustering-orange?style=flat-square)

---

## What is this?

This notebook walks through a complete **Exploratory Data Analysis (EDA)** and **K-Means clustering** of the Mall Customers dataset — the classic entry point for unsupervised machine learning.

Unlike the Titanic and Iris notebooks, there is **no label to predict**. Instead, we let the algorithm discover natural groupings in the data on its own. The output is a set of **customer segments** that a marketing team can use to tailor campaigns, offers, and loyalty programs.

The central question: **what distinct types of customers visit this mall, and how can we group them by income and spending behavior?**

---

## Dataset

| Property | Detail |
|---|---|
| Source | Public GitHub raw URL — no login needed |
| URL | `https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv` |
| Rows | 200 customers |
| Label | None — unsupervised learning |
| ML type | Clustering (K-Means) |

**Key columns:**

| Column | Type | Description |
|---|---|---|
| `CustomerID` | int | Unique identifier — dropped before clustering |
| `Gender` | categorical | Male / Female — encoded as binary |
| `Age` | numerical | Customer age |
| `Annual Income (k$)` | numerical | Annual income in thousands of USD |
| `Spending Score (1-100)` | numerical | Mall-assigned score based on spending behavior |

> There is no `y` (label) column. The algorithm finds the groups itself.

---

## What this notebook covers

**Step 1 — Data loading**
Fetch the CSV directly from a verified public GitHub URL. No Kaggle account or file upload needed. Standardize column names.

**Step 2 — Features (X) — no label y**
Explain why unsupervised learning has no target. Drop `CustomerID`. Encode `Gender` as binary. Define `X` as all remaining features.

**Step 3 — Missing values**
Confirm zero missing values. K-Means cannot handle `NaN` — this check is critical before clustering.

**Step 4 — Descriptive statistics & gender distribution**
Run `.describe()` across all features. Check gender split (56% Female, 44% Male). Discuss what distributions reveal about the customer base.

**Step 5 — Data visualization**
- KDE histograms for `age`, `income`, and `spending_score`
- Correlation heatmap — relationships between all features
- **Income vs. Spending Score scatter** — the key 2D view where 5 clusters are visible by eye before running any algorithm
- Pair plot colored by gender — explore how gender interacts with features

**Step 6 — Elbow Method (choosing k)**
Run K-Means for k = 1 to 10. Plot inertia vs. k. Identify the "elbow" at **k=5** where adding more clusters stops providing meaningful improvement. Apply `StandardScaler` first so income scale doesn't dominate.

**Step 7 — K-Means clustering (k=5)**
Fit the final model. Plot the 5 clusters with centroids on the income vs. spending scatter. Print a cluster summary table showing mean age, income, and spending score per segment.

**Step 8 — EDA report**
Structured summary of data quality, cluster profiles, and preprocessing steps applied.

---

## Key findings

- **Income vs. Spending Score** is the most revealing 2D feature combination — 5 groups are visible by eye before any algorithm runs
- **Age** has a mild negative correlation with spending score — younger customers tend to spend more relative to income
- Gender does not strongly separate clusters — income and spending are the dominant axes
- The Elbow Method confirms **k=5** as the optimal number of clusters

---

## The 5 customer segments

| Cluster | Income | Spending Score | Profile |
|---|---|---|---|
| 0 | Low | High | Young spenders — high engagement, low earnings |
| 1 | High | High | Premium customers — top loyalty program targets |
| 2 | Mid | Mid | Average customers — the largest, most typical group |
| 3 | High | Low | Careful high earners — need conversion incentives |
| 4 | Low | Low | Budget-conscious — price-sensitive segment |

---

## Preprocessing applied

- Dropped `CustomerID` — non-informative identifier
- Encoded `Gender` as binary (Male = 1, Female = 0)
- Applied `StandardScaler` before K-Means to normalize feature scales
- Used the Elbow Method to select k=5 objectively

---

## How to run

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook and paste cells in order
3. Run all — the dataset loads automatically from the URL, no file uploads needed

```bash
# All dependencies come pre-installed in Colab
pandas | numpy | matplotlib | seaborn | scikit-learn
```
