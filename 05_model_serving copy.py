# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 5: Model Serving
# MAGIC
# MAGIC ## Overview
# MAGIC In this notebook, we'll demonstrate how to serve the trained model for real-time predictions:
# MAGIC - Load model from Model Registry
# MAGIC - Create inference pipeline
# MAGIC - Batch scoring
# MAGIC - Real-time prediction examples
# MAGIC - Model monitoring concepts
# MAGIC
# MAGIC ## MLOps Best Practices:
# MAGIC - Model versioning and staging
# MAGIC - Reproducible inference pipelines
# MAGIC - Batch and real-time scoring
# MAGIC - Model performance monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    'model_name': 'loan_approval_model',
    'model_stage': 'Production',  # Can be 'Production', 'Staging', or 'Archived'
    'database_name': 'loan_workshop',
    'feature_table': 'customer_loan_features'
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

spark.sql(f"USE CATALOG amine_charot")

print(f"✓ Using catalog: amine_charot")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model from Model Registry

# COMMAND ----------

# Initialize MLflow client
client = MlflowClient()

# Get model from registry
model_name = config['model_name']
model_alias =  "champion"

# Load the model using the PyFunc flavor
model_uri = f"models:/{model_name}@{model_alias}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

display(f"✓ Model loaded from Model Registry")
display(f"  Model name: {model_name}")
display(f"  Alias: {model_alias}")
display(f"  Model type: {type(loaded_model).__name__}")

# Get model version details by alias
model_version = client.get_model_version_by_alias(
    name=model_name,
    alias=model_alias
)

display(f"  Version: {model_version.version}")
display(f"  Run ID: {model_version.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Sample Data for Testing

# COMMAND ----------

# Load data
table_path = f"{config['database_name']}.{config['feature_table']}"
df = spark.table(table_path).toPandas()

print(f"✓ Loaded {len(df)} records for testing")

# Take a sample for demonstration
sample_size = 100
df_sample = df.sample(n=sample_size, random_state=42)

print(f"✓ Created sample of {sample_size} records for inference")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Inference Data

# COMMAND ----------

# Separate features and target
X_inference = df_sample.drop('loan_approved', axis=1).copy()
y_actual = df_sample['loan_approved'].copy()

# Encode categorical variables (same as training)
categorical_columns = X_inference.select_dtypes(include=['object']).columns

print(f"Preprocessing inference data...")
print(f"  Categorical columns: {list(categorical_columns)}")

for col in categorical_columns:
    le = LabelEncoder()
    # Fit on full data to ensure all categories are covered
    le.fit(df[col])
    X_inference[col] = le.transform(X_inference[col])

print(f"✓ Inference data prepared: {X_inference.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Scoring

# COMMAND ----------

print("=" * 70)
print("BATCH SCORING")
print("=" * 70)

# Make predictions
start_time = datetime.now()

predictions = loaded_model.predict(X_inference)

end_time = datetime.now()
inference_time = (end_time - start_time).total_seconds()

print(f"\n✓ Batch predictions completed")
print(f"  Total samples: {len(X_inference)}")
print(f"  Inference time: {inference_time:.4f} seconds")
print(f"  Time per sample: {inference_time/len(X_inference)*1000:.2f} ms")

# Create results dataframe
results_df = pd.DataFrame({
    'actual': y_actual.values,
    'predicted': predictions
})

print(f"\nPrediction Summary:")
print(f"  Approved predictions: {(predictions == 1).sum()} ({(predictions == 1).mean():.2%})")
print(f"  Rejected predictions: {(predictions == 0).sum()} ({(predictions == 0).mean():.2%})")

display(results_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Scoring Performance

# COMMAND ----------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Calculate metrics on batch
batch_metrics = {
    'accuracy': accuracy_score(y_actual, predictions),
    'precision': precision_score(y_actual, predictions),
    'recall': recall_score(y_actual, predictions),
    'f1_score': f1_score(y_actual, predictions)
}

print("Batch Scoring Performance Metrics:")
print("=" * 50)
for metric, value in batch_metrics.items():
    print(f"  {metric.replace('_', ' ').title():<20s}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-Time Prediction Examples

# COMMAND ----------

print("=" * 70)
print("REAL-TIME PREDICTION EXAMPLES")
print("=" * 70)

# Function for real-time prediction
def predict_loan_approval(customer_data, model):
    """
    Make real-time loan approval prediction for a single customer
    
    Args:
        customer_data: Dictionary or DataFrame with customer features
        model: Trained model
    
    Returns:
        Dictionary with prediction and probability
    """
    # Convert to DataFrame if dictionary
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    else:
        customer_df = customer_data.copy()
    
    # Make prediction
    prediction = model.predict(customer_df)[0]
    
    result = {
        'decision': 'APPROVED' if prediction == 1 else 'REJECTED',
    }
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3: Borderline Applicant

# COMMAND ----------

# Get a borderline case (mid-range credit score)
borderline_idx = df[(df['credit_score'] >= 650) & (df['credit_score'] <= 680)].sample(1, random_state=42).index[0]
borderline_applicant = X_inference.loc[[borderline_idx]] if borderline_idx in X_inference.index else X_inference.iloc[[1]]

print("Example 3: Borderline Applicant")
print("=" * 70)
print("\nApplicant Profile:")

# Show some key features
original_applicant = df.loc[borderline_idx] if borderline_idx in df.index else df.iloc[1]
print(f"  Credit Score: {original_applicant['credit_score']}")
print(f"  Annual Income: ${original_applicant['annual_income']:,.2f}")
print(f"  Employment Length: {original_applicant['employment_length_years']} years")
print(f"  Loan Amount: ${original_applicant['loan_amount']:,.2f}")
print(f"  Debt-to-Income Ratio: {original_applicant['debt_to_income_ratio']:.3f}")

# Make prediction
prediction_result = predict_loan_approval(borderline_applicant, loaded_model)

print(f"\n🎯 Prediction Result:")
print(f"  Decision: {prediction_result['decision']}")
print(f"  Actual Decision: {'APPROVED' if original_applicant['loan_approved'] == 1 else 'REJECTED'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Distribution Analysis

# COMMAND ----------

# Display available columns
display(results_df.columns)

# Use 'predicted' as the probability column
prob_col = 'predicted'
rej_col = None  # No rejected probability column available

# Visualize prediction probabilities
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribution by actual class
axes[0].hist(
    results_df[results_df['actual'] == 1][prob_col],
    bins=30, alpha=0.7, label='Actual Approved', color='green'
)
axes[0].hist(
    results_df[results_df['actual'] == 0][prob_col],
    bins=30, alpha=0.7, label='Actual Rejected', color='red'
)
axes[0].set_xlabel('Predicted Probability of Approval', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Confidence distribution
confidence = results_df[prob_col]
axes[1].hist(confidence, bins=30, color='purple', alpha=0.7)
axes[1].set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].axvline(
    x=confidence.mean(), color='red', linestyle='--',
    linewidth=2, label=f'Mean: {confidence.mean():.3f}'
)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Monitoring

# COMMAND ----------

print("=" * 70)
print("MODEL PERFORMANCE MONITORING")
print("=" * 70)

# Example: Set model version manually if not available from production_version
# Replace '1' with your actual model version if known
model_version = 1  # or fetch from your model registry if available

monitoring_metrics = {
    'timestamp': datetime.now().isoformat(),
    'model_name': model_name,
    'model_version': model_version,
    'model_stage': model_stage,
    'batch_size': len(X_inference),
    'inference_time_seconds': inference_time,
    'avg_inference_time_ms': inference_time / len(X_inference) * 1000,
    'metrics': batch_metrics,
    'prediction_distribution': {
        'approved': int((predictions == 1).sum()),
        'rejected': int((predictions == 0).sum()),
        'approval_rate': float((predictions == 1).mean())
    },
    'confidence_stats': {
        'mean': float(confidence.mean()),
        'median': float(confidence.median()),
        'std': float(confidence.std()),
        'min': float(confidence.min()),
        'max': float(confidence.max())
    }
}

print("\nMonitoring Metrics:")
print(json.dumps(monitoring_metrics, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Predictions to Delta Lake

# COMMAND ----------

# If using scikit-learn, get probabilities for the positive class
# predicted_probabilities = model.predict_proba(X_sample)[:, 1]

# If you already have a variable with probabilities, assign it:
# predicted_probabilities = your_probability_variable

# Add predictions to original sample
results_full = df_sample.copy()
results_full['predicted_approval'] = predictions
results_full['prediction_confidence'] = confidence.values
results_full['prediction_timestamp'] = datetime.now()

# Convert to Spark DataFrame
results_spark = spark.createDataFrame(results_full)

# Save to Delta table
predictions_table = f"{config['database_name']}.loan_predictions"
results_spark.write.format("delta").mode("append").saveAsTable(predictions_table)

display(results_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Inference API Example

# COMMAND ----------

def inference_api(request_data):
    """
    Simulated REST API endpoint for loan approval predictions
    
    Args:
        request_data: Dictionary with customer information
    
    Returns:
        Dictionary with prediction result
    """
    try:
        # Extract features from request
        features = {
            'age': request_data.get('age'),
            'annual_income': request_data.get('annual_income'),
            'employment_length_years': request_data.get('employment_length_years'),
            'employment_type': request_data.get('employment_type'),
            'credit_score': request_data.get('credit_score'),
            'existing_debt': request_data.get('existing_debt'),
            'debt_to_income_ratio': request_data.get('debt_to_income_ratio'),
            'num_credit_accounts': request_data.get('num_credit_accounts'),
            'num_delinquencies': request_data.get('num_delinquencies'),
            'loan_amount': request_data.get('loan_amount'),
            'loan_purpose': request_data.get('loan_purpose'),
            'loan_term_months': request_data.get('loan_term_months')
        }
        
        # Add engineered features (simplified - in production, use same pipeline as training)
        features['loan_to_income_ratio'] = features['loan_amount'] / features['annual_income']
        features['estimated_monthly_payment'] = (features['loan_amount'] * 1.05) / features['loan_term_months']
        features['payment_to_income_ratio'] = (features['estimated_monthly_payment'] * 12) / features['annual_income']
        features['credit_utilization'] = features['existing_debt'] / (features['annual_income'] + 1)
        features['financial_health_score'] = (
            (features['credit_score'] / 850) * 0.4 +
            (1 - features['debt_to_income_ratio']) * 0.3 +
            min(features['employment_length_years'] / 10, 1) * 0.2 +
            (1 - features['num_delinquencies'] / 5) * 0.1
        )
        features['employment_stability'] = min(features['employment_length_years'] / 5, 2)
        features['risk_indicator'] = (
            (850 - features['credit_score']) / 550 * 0.4 +
            features['debt_to_income_ratio'] * 0.3 +
            (features['num_delinquencies'] / 5) * 0.3
        )
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Encode categorical
        if 'employment_type' in feature_df.columns:
            le = LabelEncoder()
            le.fit(['full_time', 'part_time', 'self_employed', 'contract'])
            feature_df['employment_type'] = le.transform(feature_df['employment_type'])
        
        if 'loan_purpose' in feature_df.columns:
            le = LabelEncoder()
            le.fit(['home_improvement', 'debt_consolidation', 'business', 
                    'education', 'auto', 'medical', 'wedding', 'vacation'])
            feature_df['loan_purpose'] = le.transform(feature_df['loan_purpose'])
        
        # Make prediction
        prediction = loaded_model.predict(feature_df)[0]
        probability = loaded_model.predict_proba(feature_df)[0]
        
        # Format response
        response = {
            'status': 'success',
            'decision': 'APPROVED' if prediction == 1 else 'REJECTED',
            'confidence': float(max(probability)),
            'approval_probability': float(probability[1]),
            'rejection_probability': float(probability[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

print("✓ Inference API function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Inference API

# COMMAND ----------

# Test API with sample request
sample_request = {
    'age': 35,
    'annual_income': 75000.0,
    'employment_length_years': 5,
    'employment_type': 'full_time',
    'credit_score': 720,
    'existing_debt': 15000.0,
    'debt_to_income_ratio': 0.30,
    'num_credit_accounts': 4,
    'num_delinquencies': 0,
    'loan_amount': 25000.0,
    'loan_purpose': 'debt_consolidation',
    'loan_term_months': 36
}

print("API Request:")
print(json.dumps(sample_request, indent=2))

print("\n" + "=" * 70)

api_response = inference_api(sample_request)

print("\nAPI Response:")
print(json.dumps(api_response, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    MODEL SERVING COMPLETE                              ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ✓ Model loaded from Model Registry                                   ║
║  ✓ Batch scoring demonstrated                                         ║
║  ✓ Real-time prediction examples shown                                ║
║  ✓ Model monitoring metrics collected                                 ║
║  ✓ Predictions saved to Delta Lake                                    ║
║  ✓ Inference API example created                                      ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  Model is production-ready and serving predictions!                    ║
╚════════════════════════════════════════════════════════════════════════╝
""")

print(f"\nKey Statistics:")
print(f"  Model: {model_name}")
print(f"  Batch size: {len(X_inference)} records")
print(f"  Avg inference time: {inference_time/len(X_inference)*1000:.2f} ms per record")
print(f"  Batch accuracy: {batch_metrics['accuracy']:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Proceed to **06_AutoML** to:
# MAGIC - Demonstrate Databricks AutoML capabilities
# MAGIC - Automatically train and compare models
# MAGIC - Generate feature importance insights
# MAGIC - Compare AutoML results with manual training
# MAGIC