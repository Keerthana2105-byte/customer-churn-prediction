
import pandas as pd
import os

# Define dataset paths
input_path = "data/raw/Telco-Customer-Churn.csv"
output_path = "data/cleaned_customer_churn.csv"

# Check dataset
if not os.path.exists(input_path):
    print("ERROR: Original dataset not found!")
    exit()

# Load dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Clean text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Create data folder if missing
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("Cleaned dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Cleaned dataset saved successfully!")
