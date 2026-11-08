# Credit Card Fraud Detection

A Machine Learning project that aims to identify fraudulent credit card transactions using classification algorithms.

## Project Overview

Credit card fraud is a major challenge in financial transactions. This project explores machine learning techniques to distinguish fraudulent transactions from genuine ones using a highly imbalanced dataset.

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 30 input features
- **Target:** Class (0 = Genuine, 1 = Fraud)

The dataset is not included in this repository. Download it from Kaggle and place it inside the `data/archive/` directory.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
│   └── figures/
├── .gitignore
└── README.md
```

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Handling Class Imbalance
4. Model Training
5. Model Evaluation
6. Model Comparison

## Evaluation Metrics

- Precision
- Recall
- F1-Score
- Confusion Matrix
- Precision-Recall AUC

## Project Status

In Progress 🚧