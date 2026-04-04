# ML EDA Notebook Series

A beginner-friendly series of three Google Colab notebooks covering the full **pre-modeling workflow** — exploratory data analysis, core ML terminology, and data preparation — across three real-world datasets.

No model training here. The focus is on understanding your data before you touch an algorithm.

---

## What's inside

| Notebook | Dataset | ML Type | Key concepts |
|---|---|---|---|
| `titanic_eda.ipynb` | Titanic | Binary Classification | Features vs. label, missing values, class imbalance, 70/30 split |
| `iris_eda.ipynb` | Iris | Multiclass Classification | Pair plots, multicollinearity, supervised vs. unsupervised, 80/20 split |
| `mall_customers_eda.ipynb` | Mall Customers | Clustering (Unsupervised) | No label, Elbow Method, K-Means, customer segmentation |

---

## Workflow covered in each notebook

```
Load data → Define X and y → Check data types → Handle missing values
→ Descriptive stats → Visualize → Train/test split → EDA report
```

---

## How to run

All notebooks run fully in **Google Colab** — no installs, no file uploads, no API keys.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload or open the notebook
3. Run all cells top to bottom

Dependencies used: `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

---

## Who this is for

Anyone learning ML who wants to build the habit of understanding their data before modeling. Each notebook is self-contained and can be read independently.
