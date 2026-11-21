
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports folder if missing
os.makedirs("reports", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

sns.set_theme(style="whitegrid")

# 1. Customer churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig("reports/churn_distribution.png")
plt.show()

# 2. Churn by contract type
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/churn_by_contract.png")
plt.show()

# 3. Monthly charges distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.savefig("reports/monthly_charges.png")
plt.show()

# 4. Churn by payment method
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Customer Churn by Payment Method")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/churn_by_payment.png")
plt.show()

# 5. Correlation heatmap for numeric columns
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.show()

print("EDA completed successfully!")
print("All graphs saved inside reports folder.")
