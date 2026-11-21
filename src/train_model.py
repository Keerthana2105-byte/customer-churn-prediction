
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset loaded:", df.shape)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Define ML models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []
best_f1 = -1
best_model = None
best_model_name = ""

# Train and evaluate models
for name, algorithm in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    # Select best model using F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name
        best_predictions = predictions

# Save model comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("reports/model_comparison.csv", index=False)

# Save best model
joblib.dump(best_model, "models/best_churn_model.pkl")

# Save confusion matrix
cm = confusion_matrix(y_test, best_predictions)
pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
).to_csv("reports/confusion_matrix.csv")

# Save classification report
report = classification_report(
    y_test,
    best_predictions,
    target_names=["No Churn", "Churn"]
)

with open("reports/classification_report.txt", "w") as file:
    file.write(report)

print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))

print("\nBest model saved successfully!")
print("Location: models/best_churn_model.pkl")
print("Evaluation reports saved inside reports folder.")
