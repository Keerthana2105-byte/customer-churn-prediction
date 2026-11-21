# Customer Churn Prediction System

## Project Overview
The Customer Churn Prediction System is a Machine Learning project that predicts whether a customer is likely to leave a telecom company.

It uses customer information, service subscriptions, contract details and billing information to identify potential customer churn.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models
Four classification algorithms were implemented and compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## Model Performance

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 73.81% | 61.36% |
| Decision Tree | 73.17% | 49.06% |
| Random Forest | 76.37% | 58.22% |
| Gradient Boosting | 80.62% | 58.95% |

Logistic Regression was selected as the final model based on its highest F1 Score.

## Features
- Customer data cleaning and preprocessing
- Exploratory Data Analysis
- Machine Learning model training
- Model comparison and evaluation
- Customer churn prediction through a Streamlit web application
- Churn probability display

## Project Structure
```text
customer-churn-prediction/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── cleaned_customer_churn.csv
├── models/
│   └── best_churn_model.pkl
├── reports/
├── src/
│   ├── inspect_data.py
│   ├── data_cleaning.py
│   ├── eda.py
│   └── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train_model.py
```

### 4. Run the Streamlit application
```bash
python -m streamlit run app/app.py
```

## Application
The application accepts customer details and displays the predicted churn status and churn probability.

## Learning Outcomes
- Data preprocessing and feature engineering
- Exploratory data analysis
- Classification algorithms
- Model evaluation and comparison
- Building an interactive ML application

## Author
Keerthana

B.Tech Computer Science and Engineering
