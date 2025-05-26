# Santander Customer Transaction Prediction Analysis

This project involves a complete data analysis and machine learning pipeline to predict whether a customer will make a specific transaction. 

## 📌 Project Overview

The objective of this project is to:

* Analyze customer transaction data
* Engineer features for improved model performance
* Build and evaluate machine learning models to predict binary outcomes

The solution leverages the **LightGBM** algorithm, which is well-suited for structured tabular data and offers robust performance with efficient training time.

---

## 📁 Dataset

The dataset consists of:

* `train.csv` — Training data with features and target labels
  🔗 [Download train.csv](https://drive.google.com/file/d/1WYAjM37tmqSTQ3RivckJUMoB9IZ3qsN9/view)

* `test.csv` — Test data without labels
  🔗 [Download test.csv](https://drive.google.com/file/d/1XQ5Ys7f_eVlJGL24Mb9umk9qAAwI0GPF/view)

To use them in Google Colab:

```python
!gdown --id 1WYAjM37tmqSTQ3RivckJUMoB9IZ3qsN9
!gdown --id 1XQ5Ys7f_eVlJGL24Mb9umk9qAAwI0GPF

import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
```

---


## 🧰 Tools & Libraries

* Python 3
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn
* LightGBM
* TQDM
* Warnings, OS, Logging, GC (for resource management)

---

## 🔍 Exploratory Data Analysis (EDA)

The analysis includes:

* Dataset shape and preview (`head()`)
* Missing value analysis (none found)
* Target class distribution
* Correlation heatmaps
* Feature distribution visualizations
* Detection of constant and duplicate features

---

## 🧪 Feature Engineering

* Removed constant features with no variance
* Identified and dropped duplicate features
* Generated new features based on:

  * Mean, min, max, standard deviation across rows
  * Count of positive/negative values
  * Sum and median of features
* Created rank-based and z-score-based features

---

## 🧠 Model Building

The main model used:

* **LightGBM Classifier**

  * 5-fold **Stratified K-Fold Cross Validation**
  * AUC score used as the evaluation metric
  * Feature importance plot generated for interpretability

---

## 📈 Results

* The model achieved a strong AUC on both train and validation sets
* Insights gained from feature importance helped understand key drivers in the data
* Feature engineering significantly improved model performance

---

## 📊 Visualizations

* Distribution plots for top features
* Correlation matrix for features
* Feature importance bar chart from LightGBM

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Place the dataset files (`train.csv`, `test.csv`) in the specified directory.
3. Run the notebook `Santander_Analysis.ipynb` in Jupyter or Colab.

---

## 📌 Project Structure

```
.
├── Santander_Analysis.ipynb   # Main analysis and modeling notebook
├── README.md                  # Project description and documentation
└── data/
    ├── train.csv              # Training data
    └── test.csv               # Test data
```

---

## 📜 License

This project is intended for educational and research purposes.

