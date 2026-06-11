# 🏦 Credit Risk Prediction using Logistic Regression (Orange Tool)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-blue)
![Tool](https://img.shields.io/badge/Tool-Orange%20Data%20Mining-orange)
![Dataset](https://img.shields.io/badge/Dataset-100%20Rows%20%7C%2012%20Features-green)
![Accuracy](https://img.shields.io/badge/Accuracy-69%25-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

-----

## 📌 Project Overview

This lab demonstrates how to build a **Credit Risk Prediction Model** using **Logistic Regression** in **Orange Data Mining** — a no-code machine learning tool.

> **Real World Scenario:** A bank wants to predict whether a loan applicant will **default** on their loan based on financial history. This helps reduce bad loans, speed up approvals, and make smarter lending decisions.

-----

## 🎯 Objective

- ✅ Build a Logistic Regression model using Orange (no-code)
- ✅ Predict loan default using real-world style customer data
- ✅ Evaluate model using AUC, Accuracy, F1, Precision, Recall, MCC
- ✅ Understand how banks use ML for credit risk analysis

-----

## 🛠️ Tool Used

**Orange Data Mining** — Free, open-source, visual machine learning tool

🔗 Download: <https://orangedatamining.com/download/>

-----

## 📂 Dataset Description

**File:** `loan default dataset.csv`
**Rows:** 100 customers | **Columns:** 12 features

|# |Column              |Type      |Description                    |
|--|--------------------|----------|-------------------------------|
|1 |CustomerID          |Numeric   |Unique customer identifier     |
|2 |Age                 |Numeric   |Customer age                   |
|3 |Annual_Income       |Numeric   |Yearly income (₹)              |
|4 |Loan_Amount         |Numeric   |Loan requested (₹)             |
|5 |Loan_Term_Months    |Numeric   |Repayment duration in months   |
|6 |Credit_Score        |Numeric   |Credit score (500–850)         |
|7 |Existing_Loans      |Numeric   |Number of active loans         |
|8 |Employment_Years    |Numeric   |Years of employment            |
|9 |Debt_to_Income_Ratio|Numeric   |Loan Amount ÷ Annual Income    |
|10|Savings_Balance     |Numeric   |Total savings (₹)              |
|11|Monthly_EMI         |Numeric   |Monthly installment amount     |
|12|**Default**         |**Target**|**1 = Default, 0 = No Default**|

-----

## 🔄 Orange Workflow — Step by Step

### Step 1: Load the File

> Drag the **File Widget** onto the canvas and load the dataset.

![File Widget](https://raw.githubusercontent.com/klathwalipm22-tech/Credit-Risk-Logistic-Regression-Orange/main/file.png)

-----

### Step 2: Select Target Column

> Use **Select Columns Widget** to set `Default` as the **Target Variable**.

![Select Columns](https://raw.githubusercontent.com/klathwalipm22-tech/Credit-Risk-Logistic-Regression-Orange/main/select%20columm.png)

-----

### Step 3: Apply Logistic Regression Model

> Add **Logistic Regression Widget** and connect it to the workflow.

![Logistic Regression](https://raw.githubusercontent.com/klathwalipm22-tech/Credit-Risk-Logistic-Regression-Orange/main/Logic%20Regrasstion.png)

-----

### Step 4: Connect All Widgets

> Final workflow: `File → Select Columns → Test and Score ← Logistic Regression`

![Full Workflow](https://raw.githubusercontent.com/klathwalipm22-tech/Credit-Risk-Logistic-Regression-Orange/main/conecting%20all%20steps.png)

-----

### Step 5: View Results

> Open **Test and Score Widget** to see model evaluation metrics.

![Results](https://raw.githubusercontent.com/klathwalipm22-tech/Credit-Risk-Logistic-Regression-Orange/main/Results.png)

-----

## 📊 Model Results

|Metric             |Score    |Interpretation                            |
|-------------------|---------|------------------------------------------|
|🔵 **AUC**          |**0.676**|Model separates defaulters reasonably well|
|🟢 **CA (Accuracy)**|**0.690**|69% of predictions are correct            |
|🟡 **F1 Score**     |**0.690**|Balanced precision & recall performance   |
|🟠 **Precision**    |**0.689**|69% correct when predicting default       |
|🔴 **Recall**       |**0.690**|Catches 69% of actual defaulters          |
|⚪ **MCC**          |**0.373**|Moderate overall model quality            |


> **Evaluation Method:** Cross Validation | Folds: **5** | Stratified: **Yes**

-----

## 💻 Dataset Generation Code (Python)

```python
import csv, random, math
random.seed(42)

rows = []
for i in range(100):
    age = random.randint(22, 60)
    income = random.randint(200000, 1500000)
    loan_amount = random.randint(50000, 800000)
    loan_term = random.choice([12, 24, 36, 48, 60])
    credit_score = random.randint(500, 850)
    existing_loans = random.randint(0, 4)
    employment_years = random.randint(0, 20)
    debt_to_income = round(loan_amount / income, 2)
    monthly_emi = round(loan_amount / loan_term, 0)
    savings = random.randint(10000, 500000)

    # Default logic: low credit + high DTI + low savings = higher default risk
    score = (credit_score - 500)/350 + (1 - debt_to_income) + (savings/500000) + (employment_years/20)
    prob = 1 / (1 + math.exp(-(score - 1.5)))
    default = 0 if random.random() < prob else 1

    rows.append([i+1, age, income, loan_amount, loan_term, credit_score,
                 existing_loans, employment_years, debt_to_income, savings, monthly_emi, default])

with open('loan_default_dataset.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['CustomerID','Age','Annual_Income','Loan_Amount','Loan_Term_Months',
                'Credit_Score','Existing_Loans','Employment_Years','Debt_to_Income_Ratio',
                'Savings_Balance','Monthly_EMI','Default'])
    w.writerows(rows)

print("Dataset created: 100 rows, 12 columns")
```

-----

## 💡 Business Understanding

Banks can use this model to:

|Use Case                     |Benefit                    |
|-----------------------------|---------------------------|
|🚫 Reject high-risk applicants|Reduce bad loan portfolio  |
|⚡ Automate approvals         |Speed up processing time   |
|📈 Data-driven decisions      |Replace gut-feeling with ML|
|💰 Reduce financial losses    |Early default detection    |

-----

## ⚠️ Limitations

- Accuracy is moderate (69%) — can improve with larger dataset
- Logistic Regression may miss complex non-linear patterns
- Model performance depends on data quality and feature engineering

-----

## 🎓 Key Concepts Learned

|Concept                |Definition                                                             |
|-----------------------|-----------------------------------------------------------------------|
|**Logistic Regression**|Predicts probability of binary outcome (Default/No Default)            |
|**AUC**                |Measures model’s ability to separate two classes (closer to 1 = better)|
|**Cross Validation**   |Splits data into 5 folds for reliable unbiased evaluation              |
|**Precision**          |Of all predicted defaults, how many were actually defaults             |
|**Recall**             |Of all actual defaults, how many did the model catch                   |
|**F1 Score**           |Harmonic mean of Precision and Recall                                  |
|**MCC**                |Overall balanced metric even with unequal class sizes                  |

-----

## 📁 Repository Structure

```
Credit-Risk-Logistic-Regression-Orange/
│
├── 📄 loan default dataset.csv       # Dataset (100 rows, 12 columns)
├── 🖼️ file.png                       # Screenshot: File widget
├── 🖼️ select columm.png              # Screenshot: Select Columns widget
├── 🖼️ Logic Regrasstion.png          # Screenshot: Logistic Regression widget
├── 🖼️ conecting all steps.png        # Screenshot: Full workflow
├── 🖼️ Results.png                    # Screenshot: Model results
├── 🖼️ test and score.png             # Screenshot: Test and Score widget
├── 🖼️ dataset.png                    # Screenshot: Dataset view
└── 📝 README.md                      # Project documentation
```

-----

## 👩‍🎓 Author

**Khushi** | MBA Applied Finance  
🏫 Chitkara University, Panchkula  
📚 Course: AI Agents & Automation Lab  
🔗 GitHub: [klathwalipm22-tech](https://github.com/klathwalipm22-tech)

-----

⭐ *If you found this helpful, give it a star!*
