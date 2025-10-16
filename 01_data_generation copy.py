# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 1: Data Generation
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook generates synthetic customer loan application data for our ML workshop.
# MAGIC We'll create realistic customer profiles with features that influence loan approval decisions.
# MAGIC
# MAGIC ## Features:
# MAGIC - **Customer Demographics**: Age, income, employment length
# MAGIC - **Credit Information**: Credit score, existing debt
# MAGIC - **Loan Details**: Loan amount, loan purpose, term
# MAGIC - **Target Variable**: Loan approval status (approved/rejected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("✓ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration parameters
config = {
    'num_samples': 10000,
    'train_test_split': 0.8,
    'catalog_name': 'amine_charot',
    'database_name': 'loan_workshop',
    'table_name': 'customer_loan_applications'
}

print(f"Configuration:")
print(f"  - Total samples: {config['num_samples']}")
print(f"  - Train/Test split: {config['train_test_split']}")
print(f"  - Database: {config['database_name']}")
print(f"  - Table: {config['table_name']}")

# COMMAND ----------

# Set the current catalog from config
spark.sql(f"USE CATALOG {config['catalog_name']}")
print(f"✓ Using catalog: {config['catalog_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation Functions

# COMMAND ----------

def generate_customer_data(num_samples):
    """
    Generate synthetic customer loan application data
    """
    
    # Loan purposes
    loan_purposes = ['home_improvement', 'debt_consolidation', 'business', 
                     'education', 'auto', 'medical', 'wedding', 'vacation']
    
    # Employment types
    employment_types = ['full_time', 'part_time', 'self_employed', 'contract']
    
    data = []
    
    for i in range(num_samples):
        # Customer demographics
        age = np.random.normal(40, 12)
        age = max(18, min(75, age))  # Constrain between 18-75
        
        # Annual income (correlated with age)
        base_income = 30000 + (age - 18) * 1500
        income = np.random.lognormal(np.log(base_income), 0.5)
        income = max(15000, min(500000, income))
        
        # Employment length (correlated with age)
        max_employment = min(age - 18, 40)
        employment_length = np.random.randint(0, max(1, max_employment))
        
        # Credit score (somewhat correlated with income and employment)
        base_credit = 600 + (income / 10000) + (employment_length * 2)
        credit_score = int(np.random.normal(base_credit, 50))
        credit_score = max(300, min(850, credit_score))
        
        # Existing debt
        debt_to_income_ratio = np.random.beta(2, 5)
        existing_debt = income * debt_to_income_ratio * np.random.uniform(0.1, 0.5)
        
        # Loan details
        loan_amount = np.random.uniform(5000, min(income * 0.5, 100000))
        loan_purpose = random.choice(loan_purposes)
        loan_term = random.choice([12, 24, 36, 48, 60])
        employment_type = random.choice(employment_types)
        
        # Number of existing accounts
        num_credit_accounts = np.random.poisson(3) + 1
        
        # Number of delinquencies (inversely correlated with credit score)
        delinquency_prob = max(0, (750 - credit_score) / 450)
        num_delinquencies = np.random.binomial(5, delinquency_prob)
        
        # Calculate loan approval probability based on features
        # Higher credit score, income, employment length = higher approval
        # Higher debt-to-income, delinquencies = lower approval
        
        approval_score = (
            (credit_score - 300) / 550 * 0.35 +  # Credit score weight
            min(income / 100000, 1) * 0.25 +      # Income weight
            min(employment_length / 10, 1) * 0.15 + # Employment weight
            (1 - debt_to_income_ratio) * 0.15 +    # Debt-to-income weight
            (1 - num_delinquencies / 5) * 0.10     # Delinquencies weight
        )
        
        # Add some randomness
        approval_score += np.random.normal(0, 0.1)
        approval_score = max(0, min(1, approval_score))
        
        # Approve if score > 0.5 (roughly 50-50 split with some variance)
        loan_approved = 1 if approval_score > 0.5 else 0
        
        # Create record
        record = {
            'customer_id': f'CUST_{i:06d}',
            'application_date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'age': int(age),
            'annual_income': round(income, 2),
            'employment_length_years': employment_length,
            'employment_type': employment_type,
            'credit_score': credit_score,
            'existing_debt': round(existing_debt, 2),
            'debt_to_income_ratio': round(debt_to_income_ratio, 3),
            'num_credit_accounts': num_credit_accounts,
            'num_delinquencies': num_delinquencies,
            'loan_amount': round(loan_amount, 2),
            'loan_purpose': loan_purpose,
            'loan_term_months': loan_term,
            'loan_approved': loan_approved
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

print("✓ Data generation function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Dataset

# COMMAND ----------

# Generate the data
print(f"Generating {config['num_samples']} customer loan applications...")
df_pandas = generate_customer_data(config['num_samples'])

print(f"\n✓ Generated {len(df_pandas)} records")
print(f"\nDataset shape: {df_pandas.shape}")
print(f"\nApproval rate: {df_pandas['loan_approved'].mean():.2%}")

# Display first few rows
display(df_pandas.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

print("Data Quality Checks:")
print("=" * 50)

# Check for missing values
missing = df_pandas.isnull().sum()
print(f"\n1. Missing Values:")
print(f"   Total missing values: {missing.sum()}")

# Check data types
print(f"\n2. Data Types:")
print(df_pandas.dtypes)

# Check value ranges
print(f"\n3. Value Ranges:")
print(f"   Age: {df_pandas['age'].min():.0f} - {df_pandas['age'].max():.0f}")
print(f"   Income: ${df_pandas['annual_income'].min():.0f} - ${df_pandas['annual_income'].max():.0f}")
print(f"   Credit Score: {df_pandas['credit_score'].min()} - {df_pandas['credit_score'].max()}")
print(f"   Loan Amount: ${df_pandas['loan_amount'].min():.2f} - ${df_pandas['loan_amount'].max():.2f}")

# Check class balance
print(f"\n4. Target Variable Balance:")
print(df_pandas['loan_approved'].value_counts())
print(f"   Approval rate: {df_pandas['loan_approved'].mean():.2%}")

print("\n✓ Data quality checks completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Summary

# COMMAND ----------

# Display statistical summary
print("Statistical Summary of Numerical Features:")
display(df_pandas.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Lake

# COMMAND ----------

# Convert to Spark DataFrame
df_spark = spark.createDataFrame(df_pandas)

# Create database if not exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS {config['database_name']}")
print(f"✓ Database '{config['database_name']}' ready")

# Save as Delta table
table_path = f"{config['database_name']}.{config['table_name']}"
df_spark.write.format("delta").mode("overwrite").saveAsTable(table_path)

print(f"✓ Data saved to Delta table: {table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Saved Data

# COMMAND ----------

# Read back the data to verify
df_verification = spark.table(table_path)

print(f"Verification:")
print(f"  - Record count: {df_verification.count()}")
print(f"  - Schema:")

df_verification.printSchema()

# Show sample records
print("\nSample records from saved table:")
display(df_verification.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics by Approval Status

# COMMAND ----------

# Compare approved vs rejected applications
print("Comparison: Approved vs Rejected Applications")
print("=" * 70)

approved_stats = df_pandas[df_pandas['loan_approved'] == 1].describe()
rejected_stats = df_pandas[df_pandas['loan_approved'] == 0].describe()

comparison_features = ['age', 'annual_income', 'credit_score', 'employment_length_years', 
                       'debt_to_income_ratio', 'num_delinquencies', 'loan_amount']

print("\nKey metrics comparison (mean values):")
for feature in comparison_features:
    approved_mean = approved_stats.loc['mean', feature]
    rejected_mean = rejected_stats.loc['mean', feature]
    print(f"  {feature:30s} | Approved: {approved_mean:12.2f} | Rejected: {rejected_mean:12.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ✓ **Data Generation Complete!**
# MAGIC
# MAGIC The synthetic customer loan dataset has been created and saved to Delta Lake.
# MAGIC
# MAGIC **Next Notebooks:**
# MAGIC 1. **02_EDA_and_Feature_Engineering**: Exploratory Data Analysis and Feature Engineering
# MAGIC 2. **03_Model_Training**: Train multiple models with MLflow tracking
# MAGIC 3. **04_Model_Evaluation**: Compare and select the best model
# MAGIC 4. **05_Model_Registry_and_Serving**: Deploy model for serving
# MAGIC 5. **06_AutoML**: Demonstrate Databricks AutoML capabilities

# COMMAND ----------

print("""
╔═══════════════════════════════════════════════════════════╗
║                  NOTEBOOK COMPLETE ✓                      ║
║                                                           ║
║  Dataset: 10,000 customer loan applications              ║
║  Location: loan_workshop.customer_loan_applications      ║
║                                                           ║
║  Ready for the next step: EDA & Feature Engineering      ║
╚═══════════════════════════════════════════════════════════╝
""")
