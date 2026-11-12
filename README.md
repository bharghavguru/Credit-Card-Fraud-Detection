# Credit Card Fraud Detection Using Machine Learning

## 1. Project Overview

Credit card fraud is a common challenge in the financial industry. One of the main difficulties in detecting it is that fraudulent transactions make up only a very small portion of all transactions.

In this project, I built a machine learning model to classify credit card transactions as genuine or fraudulent using transaction data.

I experimented with two classification algorithms: **Logistic Regression** and **Random Forest**. I compared their performance and explored techniques such as feature scaling, SMOTE, and classification threshold tuning to understand how they affect fraud detection.

## 2. Project Objectives

The main goals of this project were to:

- Explore and understand the credit card transaction dataset.
- Identify patterns in genuine and fraudulent transactions.
- Perform data cleaning and exploratory data analysis.
- Prepare the data for machine learning.
- Handle class imbalance using SMOTE.
- Train and compare Logistic Regression and Random Forest.
- Evaluate model performance using suitable classification metrics.
- Experiment with classification thresholds to improve prediction results.
- Visualize and analyze the model results.

## 3. Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains credit card transactions made by European cardholders in September 2013.

| Attribute | Description |
|---|---|
| Time | Seconds elapsed between each transaction and the first transaction |
| V1–V28 | Anonymized features obtained through PCA transformation |
| Amount | Transaction amount |
| Class | Indicates whether a transaction is genuine or fraudulent |

**Target variable:**
- `0` – Genuine transaction
- `1` – Fraudulent transaction

**Dataset details after removing duplicate records:**

- Total transactions: 283,726
- Genuine transactions: 283,253
- Fraudulent transactions: 473
- Total features: 30

One thing that stood out during the analysis was the class imbalance. Fraudulent transactions represented only around **0.17%** of the cleaned dataset, making this an interesting classification problem.

The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside `data/archive/`.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- Git and GitHub

## 5. Project Structure

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
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       └── model_comparison.png
│
├── .gitignore
└── README.md
```

## 6. Project Workflow

### Step 1: Exploratory Data Analysis

I started by loading the dataset and checking its structure and quality.

During this stage, I:

- Checked for missing values and duplicate records.
- Analyzed the distribution of genuine and fraudulent transactions.
- Examined transaction amounts and feature distributions.
- Explored correlations between features.
- Created visualizations to understand the dataset.

This helped me understand how imbalanced the data was before moving on to model development.

### Step 2: Data Preprocessing

Before training the models, I prepared the dataset by:

- Removing duplicate records.
- Separating input features and the target variable.
- Splitting the data into training and testing sets using stratified sampling.
- Applying `StandardScaler` to the `Time` and `Amount` features.
- Using SMOTE to handle the imbalance between genuine and fraudulent transactions.

The dataset was split into:

- Training set: 80%
- Testing set: 20%

An important part of this process was applying SMOTE **only to the training data**. The test set was kept unchanged so that the final evaluation would reflect the original class distribution.

### Step 3: Model Training

I trained two classification models to compare their performance.

**1. Logistic Regression**

I used Logistic Regression as the baseline model. It helped establish an initial performance result and showed how a basic classification algorithm handles an imbalanced dataset.

**2. Random Forest Classifier**

I also trained a Random Forest model, which combines predictions from multiple decision trees.

The model was configured with the following parameters:

- Number of estimators: 100
- Maximum depth: 12
- Minimum samples per leaf: 2
- Random state: 42

### Step 4: Feature Importance Analysis

After training the Random Forest model, I examined its feature importance scores to understand which input features contributed most to its predictions.

Some of the important features were:

- V14
- V10
- V12
- V17
- V4

This gave me a better understanding of which features the model relied on most during classification.

### Step 5: Classification Threshold Tuning

Initially, I used the default classification threshold of 0.5. However, I wanted to check whether changing the threshold could improve the model's performance.

I tested different thresholds using a separate validation dataset and selected **0.6** based on the highest F1-score among the values tested.

I then used this threshold for the final Random Forest evaluation on the untouched test dataset.

### Step 6: Model Evaluation

I evaluated both models using multiple metrics rather than relying only on accuracy.

The metrics included:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC (Average Precision)

These metrics helped me understand how well the models detected fraudulent transactions and how many genuine transactions they incorrectly flagged.

## 7. Model Performance Comparison

After training both models, I compared their results on the test dataset.

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.37% | 99.92% |
| Precision | 5.00% | 76.84% |
| Recall | 87.00% | 76.84% |
| F1-score | 10.00% | 76.84% |
| ROC-AUC | 96.19% | 97.89% |
| PR-AUC | 67.69% | 79.54% |

**A note about the comparison:** Logistic Regression was evaluated at the default threshold of 0.5, while Random Forest used the tuned threshold of 0.6. So, the classification metrics are not a direct comparison at the same threshold.

### What I observed

- Logistic Regression detected more fraudulent transactions, but it also produced a large number of false alarms.
- Random Forest achieved much better precision and F1-score.
- Random Forest also performed better in ROC-AUC and PR-AUC.
- In this experiment, Random Forest provided a better balance between detecting fraud and avoiding unnecessary alerts.

## 8. Final Random Forest Evaluation

I selected Random Forest as the final model based on its overall performance in this experiment.

The model was evaluated on the original, imbalanced test dataset using the selected threshold of 0.6.

| Metric | Final Result |
|---|---:|
| Accuracy | 99.92% |
| Precision | 76.84% |
| Recall | 76.84% |
| F1-score | 76.84% |
| ROC-AUC | 97.89% |
| PR-AUC | 79.54% |

### Final Confusion Matrix

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,629 | 22 |
| Actual Fraud | 22 | 73 |

### Understanding the results

- **56,629** genuine transactions were correctly classified.
- **22** genuine transactions were incorrectly flagged as fraudulent.
- **73** fraudulent transactions were correctly detected.
- **22** fraudulent transactions were missed by the model.

Out of the 95 fraudulent transactions in the test dataset, the model correctly identified 73.

## 9. Model Evaluation Visualizations

I created the following visualizations to make it easier to understand and compare the model results.

### 9.1 Confusion Matrix

The confusion matrix gives a breakdown of correct and incorrect predictions for both classes.

![Confusion Matrix](reports/figures/confusion_matrix.png)

### 9.2 ROC Curve

The ROC curve shows how the true positive rate and false positive rate change across different classification thresholds.

**ROC-AUC: 0.9789**

![ROC Curve](reports/figures/roc_curve.png)

### 9.3 Precision-Recall Curve

This curve shows the relationship between precision and recall at different thresholds. It is particularly useful for evaluating a dataset where fraudulent transactions are rare.

**Average Precision: 0.7954**

![Precision-Recall Curve](reports/figures/precision_recall_curve.png)

### 9.4 Model Comparison

This chart compares Logistic Regression and Random Forest across the six evaluation metrics used in this project.

![Model Comparison](reports/figures/model_comparison.png)

## 10. Key Learnings

While working on this project, I gained practical experience with:

- Exploring and understanding a real-world dataset.
- Cleaning data and preparing it for machine learning.
- Handling highly imbalanced datasets using SMOTE.
- Training classification models with Scikit-learn.
- Understanding the differences between Logistic Regression and Random Forest.
- Evaluating models using precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Analyzing feature importance.
- Tuning classification thresholds.
- Avoiding data leakage during preprocessing and model training.
- Documenting and managing a machine learning project using Git and GitHub.

## 11. Conclusion

This project gave me an opportunity to work on a real-world classification problem where accuracy alone is not enough to judge a model's performance.

I implemented and compared Logistic Regression and Random Forest, and experimented with SMOTE and threshold tuning to improve fraud detection.

In my experiments, Random Forest achieved the better overall balance, with an F1-score of **76.84%** and ROC-AUC of **97.89%** on the test dataset.

There is still room to improve the model, especially in detecting more fraudulent transactions while keeping false alarms under control.

### Future Improvements

- Experiment with additional classification algorithms.
- Perform hyperparameter tuning.
- Explore other techniques for handling class imbalance.
- Improve fraud detection recall while controlling false positives.

---

**Project Type:** Machine Learning | Classification | Fraud Detection

**Repository:** [Credit Card Fraud Detection](https://github.com/bharghavguru/Credit-Card-Fraud-Detection)