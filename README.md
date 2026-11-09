# Credit Card Fraud Detection

A Machine Learning project that detects fraudulent credit card transactions using classification algorithms. This project focuses on handling highly imbalanced data, comparing machine learning models, and improving fraud detection through classification threshold tuning.

## Project Overview

Credit card fraud is a major challenge in digital financial transactions. Detecting fraudulent transactions is difficult because genuine transactions significantly outnumber fraudulent ones.

This project uses machine learning techniques to classify transactions as genuine or fraudulent.

The main objective is to build and evaluate classification models that can identify fraudulent transactions while minimizing false alarms.

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Input Features:** 30
- **Target Variable:** `Class`
  - `0` – Genuine transaction
  - `1` – Fraudulent transaction

The dataset contains:
- `Time`: Time elapsed since the first transaction.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized PCA-transformed features.
- `Class`: Transaction classification label.

The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside `data/archive/`.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── archive/
│       └── creditcard.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── .gitignore
└── README.md
```

The dataset is excluded from GitHub using `.gitignore`.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Loaded and explored the credit card transaction dataset.
- Identified missing values and duplicate records.
- Analyzed genuine and fraudulent transaction distributions.
- Studied feature correlations and transaction amounts.
- Visualized important patterns in the dataset.

### 2. Data Preprocessing

- Removed duplicate records.
- Separated input features and target labels.
- Split the dataset into training and testing sets using stratified sampling.
- Applied StandardScaler to `Time` and `Amount`.
- Used SMOTE to handle class imbalance in the training dataset.

SMOTE was applied only to the training data to avoid data leakage. The test dataset retained its original class distribution.

### 3. Model Training

Two classification algorithms were implemented:

- Logistic Regression
- Random Forest Classifier

### 4. Feature Importance Analysis

Random Forest feature importance was used to identify the features that contributed most to the model's predictions.

The most important features included V14, V10, V12, V17 and V4.

### 5. Classification Threshold Tuning

Different classification thresholds were evaluated using a separate validation dataset.

The threshold of 0.6 achieved the highest F1 score among the tested thresholds.

### 6. Model Evaluation

The final Random Forest model was evaluated on the untouched test dataset using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

ROC and Precision-Recall curves were also generated.

## Model Performance Comparison

The following results were obtained from the test dataset.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.37% | 99.92% |
| Precision | 5.00% | 76.84% |
| Recall | 87.00% | 76.84% |
| F1 Score | 0.10 | 0.7684 |
| ROC-AUC | 0.9619 | 0.9789 |
| PR-AUC | 0.6769 | 0.7954 |

**Note:** Logistic Regression was evaluated at a classification threshold of 0.5, while the final Random Forest model used a threshold of 0.6.

## Final Random Forest Results

The final Random Forest model achieved the following results:

- **Accuracy:** 99.92%
- **Precision:** 76.84%
- **Recall:** 76.84%
- **F1 Score:** 76.84%
- **ROC-AUC:** 0.9789
- **PR-AUC:** 0.7954

### Confusion Matrix

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,629 | 22 |
| Actual Fraud | 22 | 73 |

The model correctly detected 73 out of 95 fraudulent transactions.

## Key Learnings

Through this project, I gained practical experience in:

- Exploratory Data Analysis using Pandas and Matplotlib.
- Data preprocessing and feature scaling.
- Handling imbalanced datasets using SMOTE.
- Implementing Logistic Regression and Random Forest.
- Evaluating classification models using multiple metrics.
- Understanding precision and recall trade-offs.
- Classification threshold tuning.
- Visualizing model performance using ROC and Precision-Recall curves.
- Managing machine learning projects using Git and GitHub.

## Conclusion

Random Forest performed better overall than Logistic Regression in this experiment, achieving higher precision, F1 score, ROC-AUC and PR-AUC.

The project demonstrates how machine learning techniques can be applied to credit card fraud detection, particularly when working with highly imbalanced datasets.

Further improvements can explore additional classification algorithms, hyperparameter optimization and more advanced imbalance-handling techniques.

## Project Status

**Status:** In Progress

More experiments and improvements are planned.
