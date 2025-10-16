# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 2: EDA & Feature Engineering
# MAGIC
# MAGIC ## Overview
# MAGIC In this notebook, we'll:
# MAGIC - Perform exploratory data analysis (EDA)
# MAGIC - Visualize key patterns and relationships
# MAGIC - Engineer new features to improve model performance
# MAGIC - Prepare data for machine learning
# MAGIC
# MAGIC ## Goals:
# MAGIC - Understand data distributions and correlations
# MAGIC - Identify important features for loan approval
# MAGIC - Create engineered features
# MAGIC - Save processed data for model training

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")

# COMMAND ----------

spark.sql(f"USE CATALOG amine_charot")

print(f"✓ Using catalog: amine_charot")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data from Delta Lake

# COMMAND ----------

# Configuration
database_name = 'loan_workshop'
table_name = 'customer_loan_applications'
table_path = f"{database_name}.{table_name}"

# Load data
df_spark = spark.table(table_path)
df = df_spark.toPandas()

print(f"✓ Loaded {len(df)} records from {table_path}")
print(f"\nDataset shape: {df.shape}")
print(f"Approval rate: {df['loan_approved'].mean():.2%}")

display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis (EDA)
# MAGIC
# MAGIC ### 1. Target Variable Distribution

# COMMAND ----------

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
df['loan_approved'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Loan Approval Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Approved', 'Rejected'], rotation=0)

# Pie chart
colors = ['#e74c3c', '#2ecc71']
df['loan_approved'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                         colors=colors, labels=['Approved', 'Rejected'])
axes[1].set_title('Loan Approval Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(f"Approval Statistics:")
print(f"  - Approved: {(df['loan_approved'] == 1).sum()} ({df['loan_approved'].mean():.2%})")
print(f"  - Rejected: {(df['loan_approved'] == 0).sum()} ({(1 - df['loan_approved'].mean()):.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Numerical Features Distribution

# COMMAND ----------

# Distribution of key numerical features
numerical_features = ['age', 'annual_income', 'credit_score', 'employment_length_years', 
                      'debt_to_income_ratio', 'loan_amount']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    axes[idx].hist(df[feature], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontweight='bold')
    axes[idx].set_xlabel(feature.replace("_", " ").title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Feature Analysis by Approval Status

# COMMAND ----------

# Box plots comparing approved vs rejected
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

comparison_features = ['credit_score', 'annual_income', 'age', 
                       'employment_length_years', 'debt_to_income_ratio', 'loan_amount']

for idx, feature in enumerate(comparison_features):
    df.boxplot(column=feature, by='loan_approved', ax=axes[idx])
    axes[idx].set_title(f'{feature.replace("_", " ").title()} by Loan Status', fontweight='bold')
    axes[idx].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
    axes[idx].set_ylabel(feature.replace("_", " ").title())
    axes[idx].get_figure().suptitle('')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Correlation Analysis

# COMMAND ----------

# Correlation heatmap
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Show strongest correlations with target
print("Strongest correlations with Loan Approval:")
print("=" * 50)
target_corr = correlation_matrix['loan_approved'].sort_values(ascending=False)
for feature, corr_value in target_corr.items():
    if feature != 'loan_approved':
        print(f"  {feature:30s}: {corr_value:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Categorical Features Analysis

# COMMAND ----------

# Analyze categorical features
categorical_features = ['employment_type', 'loan_purpose']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Employment Type
employment_approval = df.groupby('employment_type')['loan_approved'].agg(['mean', 'count'])
employment_approval['mean'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Approval Rate by Employment Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Employment Type')
axes[0].set_ylabel('Approval Rate')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Loan Purpose
loan_purpose_approval = df.groupby('loan_purpose')['loan_approved'].agg(['mean', 'count'])
loan_purpose_approval['mean'].plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black')
axes[1].set_title('Approval Rate by Loan Purpose', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Loan Purpose')
axes[1].set_ylabel('Approval Rate')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nDetailed Breakdown:")
print("\nBy Employment Type:")
print(employment_approval)
print("\nBy Loan Purpose:")
print(loan_purpose_approval)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Credit Score Analysis

# COMMAND ----------

# Credit score analysis by approval status
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Distribution by approval status
df[df['loan_approved'] == 1]['credit_score'].hist(bins=30, alpha=0.7, label='Approved', 
                                                    color='green', ax=axes[0])
df[df['loan_approved'] == 0]['credit_score'].hist(bins=30, alpha=0.7, label='Rejected', 
                                                    color='red', ax=axes[0])
axes[0].set_title('Credit Score Distribution by Approval Status', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Credit Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Credit score bins
df['credit_score_bin'] = pd.cut(df['credit_score'], 
                                 bins=[0, 580, 670, 740, 800, 850], 
                                 labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

credit_bin_approval = df.groupby('credit_score_bin')['loan_approved'].agg(['mean', 'count'])
credit_bin_approval['mean'].plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Approval Rate by Credit Score Category', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Credit Score Category')
axes[1].set_ylabel('Approval Rate')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nApproval Rate by Credit Score Category:")
print(credit_bin_approval)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC
# MAGIC Create new features that might improve model performance

# COMMAND ----------

# Create a copy for feature engineering
df_fe = df.copy()

print("Creating engineered features...")

# 1. Loan to Income Ratio
df_fe['loan_to_income_ratio'] = df_fe['loan_amount'] / df_fe['annual_income']

# 2. Monthly Payment Estimate (simple interest calculation)
df_fe['estimated_monthly_payment'] = (df_fe['loan_amount'] * 1.05) / df_fe['loan_term_months']

# 3. Payment to Income Ratio
df_fe['payment_to_income_ratio'] = (df_fe['estimated_monthly_payment'] * 12) / df_fe['annual_income']

# 4. Credit Utilization Score (normalized)
df_fe['credit_utilization'] = df_fe['existing_debt'] / (df_fe['annual_income'] + 1)

# 5. Financial Health Score (composite)
df_fe['financial_health_score'] = (
    (df_fe['credit_score'] / 850) * 0.4 +
    (1 - df_fe['debt_to_income_ratio']) * 0.3 +
    np.minimum(df_fe['employment_length_years'] / 10, 1) * 0.2 +
    (1 - df_fe['num_delinquencies'] / 5) * 0.1
)

# 6. Age group
df_fe['age_group'] = pd.cut(df_fe['age'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])

# 7. Income bracket
df_fe['income_bracket'] = pd.cut(df_fe['annual_income'], 
                                  bins=[0, 30000, 50000, 75000, 100000, 1000000],
                                  labels=['Low', 'Medium', 'High', 'Very High', 'Elite'])

# 8. Loan amount category
df_fe['loan_size'] = pd.cut(df_fe['loan_amount'], 
                             bins=[0, 15000, 30000, 50000, 100000],
                             labels=['Small', 'Medium', 'Large', 'Very Large'])

# 9. Employment stability score
df_fe['employment_stability'] = np.minimum(df_fe['employment_length_years'] / 5, 2)

# 10. Risk score (inverse of approval likelihood)
df_fe['risk_indicator'] = (
    (850 - df_fe['credit_score']) / 550 * 0.4 +
    df_fe['debt_to_income_ratio'] * 0.3 +
    (df_fe['num_delinquencies'] / 5) * 0.3
)

print(f"✓ Created {len(df_fe.columns) - len(df.columns)} new features")
print(f"\nNew features:")
new_features = [col for col in df_fe.columns if col not in df.columns]
for feature in new_features:
    print(f"  - {feature}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Engineered Features

# COMMAND ----------

# Visualize some engineered features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loan to Income Ratio
df_fe.boxplot(column='loan_to_income_ratio', by='loan_approved', ax=axes[0, 0])
axes[0, 0].set_title('Loan to Income Ratio by Approval Status', fontweight='bold')
axes[0, 0].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
axes[0, 0].get_figure().suptitle('')

# Financial Health Score
df_fe.boxplot(column='financial_health_score', by='loan_approved', ax=axes[0, 1])
axes[0, 1].set_title('Financial Health Score by Approval Status', fontweight='bold')
axes[0, 1].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
axes[0, 1].get_figure().suptitle('')

# Payment to Income Ratio
df_fe.boxplot(column='payment_to_income_ratio', by='loan_approved', ax=axes[1, 0])
axes[1, 0].set_title('Payment to Income Ratio by Approval Status', fontweight='bold')
axes[1, 0].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
axes[1, 0].get_figure().suptitle('')

# Risk Indicator
df_fe.boxplot(column='risk_indicator', by='loan_approved', ax=axes[1, 1])
axes[1, 1].set_title('Risk Indicator by Approval Status', fontweight='bold')
axes[1, 1].set_xlabel('Loan Status (0=Rejected, 1=Approved)')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance - Correlation with Target

# COMMAND ----------

# Calculate correlation of all numerical features with target
numerical_cols = df_fe.select_dtypes(include=[np.number]).columns
feature_correlations = df_fe[numerical_cols].corrwith(df_fe['loan_approved']).sort_values(ascending=False)

# Visualize
plt.figure(figsize=(12, 8))
feature_correlations[feature_correlations.index != 'loan_approved'].plot(kind='barh', color='steelblue')
plt.title('Feature Correlation with Loan Approval', fontsize=16, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("Top 10 Features Correlated with Loan Approval:")
print("=" * 60)
for idx, (feature, corr) in enumerate(feature_correlations.head(11).items(), 1):
    if feature != 'loan_approved':
        print(f"{idx:2d}. {feature:35s}: {corr:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Machine Learning

# COMMAND ----------

# Remove temporary binning columns (we'll use the numerical versions for ML)
columns_to_drop = ['credit_score_bin', 'age_group', 'income_bracket', 'loan_size', 'customer_id', 'application_date']
df_ml = df_fe.drop(columns=columns_to_drop, errors='ignore')

print(f"Dataset prepared for ML:")
print(f"  - Total features: {len(df_ml.columns) - 1}")
print(f"  - Total samples: {len(df_ml)}")
print(f"\nFeatures for ML:")
feature_cols = [col for col in df_ml.columns if col != 'loan_approved']
for idx, col in enumerate(feature_cols, 1):
    print(f"  {idx:2d}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Processed Data

# COMMAND ----------

# Convert to Spark DataFrame
df_spark_processed = spark.createDataFrame(df_ml)

# Save as new Delta table
processed_table = 'customer_loan_features'
table_path_processed = f"{database_name}.{processed_table}"

df_spark_processed.write.format("delta").mode("overwrite").saveAsTable(table_path_processed)

print(f"✓ Processed data saved to: {table_path_processed}")
print(f"  - Total records: {df_spark_processed.count()}")
print(f"  - Total features: {len(df_spark_processed.columns)}")

# Verify
display(spark.table(table_path_processed).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights Summary

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                        KEY INSIGHTS FROM EDA                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  1. CREDIT SCORE: Strong positive correlation with loan approval      ║
║     → Higher credit scores significantly increase approval chances    ║
║                                                                        ║
║  2. FINANCIAL HEALTH: Engineered financial health score shows         ║
║     strong predictive power                                           ║
║                                                                        ║
║  3. DEBT-TO-INCOME RATIO: Negative correlation with approval          ║
║     → Lower ratios preferred                                          ║
║                                                                        ║
║  4. DELINQUENCIES: Strong negative impact on approval                 ║
║     → Payment history is crucial                                      ║
║                                                                        ║
║  5. EMPLOYMENT: Longer employment history correlates with approval    ║
║                                                                        ║
║  6. INCOME: Higher income increases approval likelihood               ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  ✓ Data processed and ready for model training                        ║
║  ✓ Engineered 10 new features                                         ║
║  ✓ Saved to: loan_workshop.customer_loan_features                     ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Proceed to **03_Model_Training** to:
# MAGIC - Train multiple ML models with different algorithms
# MAGIC - Use MLflow for experiment tracking
# MAGIC - Perform hyperparameter tuning
# MAGIC - Compare model performances
# MAGIC