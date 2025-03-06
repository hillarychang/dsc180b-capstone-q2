# DSC Capstone Q2 Report

## Authors
Jevan Chahal ([j2chahal@ucsd.edu](mailto:j2chahal@ucsd.edu))  
Hillary Chang ([hic001@ucsd.edu](mailto:hic001@ucsd.edu))  
Kurumi Kaneko ([kskaneko@ucsd.edu](mailto:kskaneko@ucsd.edu))  
Kevin Wong ([kew024@ucsd.edu](mailto:kew024@ucsd.edu))  
Brian Duke ([brian.duke@prismdata.com](mailto:brian.duke@prismdata.com))  
Kyle Nero ([kyle.nero@prismdata.com](mailto:kyle.nero@prismdata.com))  

---

## Abstract
The process of capturing what makes a creditor trustworthy is especially vital within bank data due to the guidelines and ethics governing its usability. Although the quantity of data is massive, only a few features are explicitly useful in machine learning, raising the question of how we should measure customer trustworthiness toward creditors. Our methodology details refining bank data using **Natural Language Processing (NLP)**, estimating individual income based on bank data alone, and measuring **creditworthiness** efficiently and accurately.

**Code Repository:** [GitHub](https://github.com/hillarychang/dsc180b-capstone-q2)

---

## Table of Contents
- [Introduction](#introduction)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Literature Review](#literature-review)
- [References](#references)
- [Appendix](#appendix)

---

## Introduction
**Access to credit is crucial for financial stability**, yet traditional credit scoring models often exclude individuals with limited credit history. The **"Cash Score" project** aims to address this issue by utilizing transaction data to evaluate financial behaviors rather than just historical credit data. Our goal is to provide a more equitable scoring system benefiting both consumers and financial institutions.

---

## Data Description
We utilized multiple datasets providing **consumer transaction details, account balances, and delinquency indicators**:

- **`q2-ucsd-consDF.pqt`**: Contains consumer attributes like `consumer_id`, `credit_score`, and `DQ_target` (delinquency indicator).
- **`q2-ucsd-acctDF.pqt`**: Includes account-level data such as `consumer_id`, `account_id`, `balance_date`, and `balance`.
- **`q2-ucsd-trxnDF.pqt`**: Captures transactional details including `category`, `amount`, `credit_or_debit`, and `posted_date`.
- **`categories.csv`**: Maps transaction categories like Rent, Groceries, and Entertainment.

### Sample Data
**Consumer Data Sample:**  
![Consumer Data](report/figure/consumer_df.jpeg)

**Transactions Sample:**  
![Transactions](report/figure/transactions_df.jpeg)

---

## Methodology

### Exploratory Data Analysis
- Identified differences in **transaction patterns** between delinquent and non-delinquent consumers.
- Examined **seasonal trends, payday effects, and spending fluctuations**.
- Estimated **income using recurring transactions**.
- Analyzed **account fees, Buy-Now-Pay-Later (BNPL) transactions, and overdrafts**.

### Balance Trends Analysis
#### **Delinquent Consumers**
![Balance Trends (Delinquent)](report/figure/balance_delinquent.png)

#### **Non-Delinquent Consumers**
![Balance Trends (Non-Delinquent)](report/figure/balance_non_delinquent.png)

---

### Feature Engineering
Key engineered features for predicting delinquency:

- **Balance Trends**: Negative balance ratio, payday effects.
- **Transaction-Based Features**: Credit vs. debit transaction volume.
- **Account Types**: Presence of savings accounts, overdraft count.
- **Spending Balance Ratio**: 
  ![Spending Balance Ratio](report/figure/spending_balance_ratio.png)

---

## Model Training & Evaluation
### Models Trained
- **Logistic Regression**: Baseline model.
- **Random Forest**: Captures non-linear financial relationships.
- **XGBoost**: Optimized for structured financial data.
- **LightGBM**: Efficient with categorical features.
- **Neural Networks**: Captures complex spending patterns.
- **Balanced RF & CatBoost**: Handle class imbalance.

### Model Performance Comparison
| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score | Training | Prediction
|--------|---------|-----------|------------|---------|------------|-----------|------------|
| HistGB | 0.8221 | **0.9031** | **0.8739** | **0.9031** | **0.8814** | 2.5050 | 0.000019 |
| Weighted Ensemble | **0.8301** | 0.9006 | 0.8728 | 0.9006 | 0.8813 | **0.0010** | **0.000001** |
| XGBoost | 0.8232 | 0.8962 | 0.8677 | 0.8962 | 0.8778 | 2.0606 | 0.000006 |
| CatBoost | 0.8212 | 0.8892 | 0.8707 | 0.8892 | 0.8785 | 3.0342 | 0.000004 |
| Balanced RF | 0.8144 | 0.8982 | 0.8703 | 0.8982 | 0.8797 | 26.2355 | 0.000064 |

### ROC-AUC Curve
![AUC-ROC Comparison](report/figure/auc_roc_all_models.png)

---

## Key Findings
- **Balance Trends Matter**: High overdraft usage and frequent negative balances correlate with **delinquency**.
- **Spending Patterns Predict Delinquency**: Categories like **BNPL transactions and account fees** are key indicators.
- **Income Stability is Crucial**: Variability in paycheck deposits signals delinquency risk.

---

## Conclusion
- **HistGB and Balanced RF** emerged as the **top models**.
- Developing a **Cash Score** based on transactional data will enhance creditworthiness prediction.
- Next steps include **refining interpretability, reason codes, and stakeholder presentation.**

---

## Literature Review
### **Transformers in Finance**
- **Vaswani et al. (2017)**: *"Attention is All You Need"* introduced **Transformers**, enabling **advanced NLP** for transaction categorization.
- **Radford et al. (2019)**: GPT-2 demonstrated **unsupervised multitask learning**, applicable to bank memo classification.
- **DGHNL Model (2020)**: Hybrid deep-learning model for **credit scoring improvements**.

---

## References
- Vaswani, A. et al. (2017). *Attention is All You Need.*
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners.*
- Plawiak, P. et al. (2020). *Deep Genetic Hierarchical Network of Learners (DGHNL) for Credit Scoring.*

---

## Appendix
- **Project Proposal:** [Google Drive Link](https://drive.google.com/file/d/1G-DzwBNvGlgd32JJwjMDrFrnr5IMGV8t/view?usp=sharing)

## Contributions
| Name | Contributions |
|------|--------------|
| **Hillary Chang** | EDA, feature engineering, report, website |
| **Kevin Wong** | EDA, graphs, model testing |
| **Kurumi Kaneko** | EDA, graphs, report writing |
| **Jevan Chahal** | EDA, feature engineering |

Everyone contributed equally to the project and met all deadlines.

---
