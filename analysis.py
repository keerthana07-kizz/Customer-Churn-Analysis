import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Churn count
print("\nChurn Count:")
print(df["Churn"].value_counts())

# Churn percentage
churn_rate = (df["Churn"] == "Yes").mean() * 100
print("\nChurn Rate:", round(churn_rate, 2), "%")
# Churn by Plan
print("\nChurn by Plan:")

plan_churn = pd.crosstab(
    df["Plan"],
    df["Churn"],
    normalize="index"
) * 100

print(plan_churn)
# Churn by City
print("\nChurn by City:")

city_churn = pd.crosstab(
    df["City"],
    df["Churn"],
    normalize="index"
) * 100

print(city_churn)
# Average tenure by churn status
print("\nAverage Tenure by Churn Status:")

tenure_churn = df.groupby("Churn")["Tenure_Months"].mean()

print(tenure_churn)
# Average support calls by churn status
print("\nAverage Support Calls by Churn Status:")

support_churn = df.groupby("Churn")["Support_Calls"].mean()

print(support_churn)
# High-risk customers
high_risk = df[
    (df["Churn"] == "Yes") &
    (df["Support_Calls"] >= 5)
]

print("\nHigh-Risk Customers:")
print(high_risk[
    ["Customer_ID", "City", "Plan", "Support_Calls", "Tenure_Months"]
])
# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
# Churn by Plan
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Plan", hue="Churn")
plt.title("Customer Churn by Plan")
plt.xlabel("Plan")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
# Churn by City
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="City", hue="Churn")
plt.title("Customer Churn by City")
plt.xlabel("City")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
# Churn by Gender
print("\nChurn by Gender:")

gender_churn = pd.crosstab(
    df["Gender"],
    df["Churn"]
)

print(gender_churn)
# Create Age Groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[18, 25, 35, 50],
    labels=["18-25", "26-35", "36-50"]
)

print("\nChurn by Age Group:")

age_churn = pd.crosstab(
    df["Age_Group"],
    df["Churn"]
)

print(age_churn)
# Average Monthly Charges by Churn
print("\nAverage Monthly Charges by Churn:")

charges_churn = df.groupby("Churn")["Monthly_Charges"].mean()

print(charges_churn)
# Churn by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender", hue="Churn")
plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
# Churn by Age Group
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Age_Group", hue="Churn")
plt.title("Customer Churn by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
# Monthly Charges by Churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="Monthly_Charges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()