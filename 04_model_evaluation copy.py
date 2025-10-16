# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 4: Model Evaluation & Selection
# MAGIC
# MAGIC ## Overview
# MAGIC In this notebook, we'll:
# MAGIC - Compare all trained models from MLflow
# MAGIC - Perform comprehensive evaluation
# MAGIC - Select the best model (Champion)
# MAGIC - Prepare for model registry and deployment
# MAGIC
# MAGIC ## Key Activities:
# MAGIC - Load all experiments from MLflow
# MAGIC - Compare performance metrics
# MAGIC - Visualize model comparisons
# MAGIC - Select champion model
# MAGIC - Register champion model to Model Registry

# COMMAND ----------

# Databricks notebook source
# MAGIC %md
# MAGIC # Fix Model Signatures (Utility Notebook)
# MAGIC
# MAGIC ## Purpose
# MAGIC This utility notebook helps fix models that were logged **without signatures**.
# MAGIC Unity Catalog requires all models to have signatures for registration.
# MAGIC
# MAGIC ## When to Use This
# MAGIC - If you get an error: "Model passed for registration did not contain any signature metadata"
# MAGIC - If you have already trained models from notebook 03 without signatures
# MAGIC - Before running notebook 04 (Model Evaluation)
# MAGIC
# MAGIC ## What This Does
# MAGIC - Loads existing models from MLflow
# MAGIC - Re-logs them with proper signatures
# MAGIC - Preserves all metrics and parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("✓ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    'experiment_name': '/Users/loan_approval_models',
    'database_name': 'loan_workshop',
    'feature_table': 'customer_loan_features'
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data (for signature inference)

# COMMAND ----------

# Load data
table_path = f"{config['database_name']}.{config['feature_table']}"
df = spark.table(table_path).toPandas()

# Prepare features
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data (to match training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data loaded: {len(X)} samples, {len(X.columns)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Models Without Signatures

# COMMAND ----------
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
parent_dir = f"/Users/{user}"
experiment_path = f"{parent_dir}/testexperiment"

# Set experiment
mlflow.set_experiment(experiment_path)
client = MlflowClient()

# Get experiment
experiment = mlflow.get_experiment_by_name(experiment_path)
experiment_id = experiment.experiment_id

# Get all runs with "best_model" tag
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.best_model = 'true'"
)

print(f"Found {len(runs)} runs with 'best_model' tag")
print("\nRuns to fix:")
for idx, row in runs.iterrows():
    run_id = row['run_id']
    algorithm = row.get('tags.algorithm', 'unknown')
    print(f"  - {algorithm}: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-log Models with Signatures

# COMMAND ----------

print("=" * 70)
print("RE-LOGGING MODELS WITH SIGNATURES")
print("=" * 70)

new_run_ids = {}

for idx, row in runs.iterrows():
    original_run_id = row['run_id']
    algorithm = row.get('tags.algorithm', 'unknown')
    model_type = row.get('tags.model_type', 'unknown')
    
    print(f"\nProcessing: {algorithm} (run: {original_run_id[:8]}...)")
    
    try:
        # Load the original model
        model_uri = f"runs:/{original_run_id}/model"
        
        # Try to load with appropriate flavor
        if 'xgboost' in algorithm.lower():
            model = mlflow.xgboost.load_model(model_uri)
        elif 'lightgbm' in algorithm.lower():
            model = mlflow.lightgbm.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)
        
        print(f"  ✓ Model loaded successfully")
        
        # Create new run with signature
        with mlflow.start_run(run_name=f"{algorithm}_with_signature") as new_run:
            # Infer signature from data
            # Note: For scaled models (like logistic regression), you might need scaled data
            predictions = model.predict(X_train[:100])  # Use small sample for signature
            signature = infer_signature(X_train[:100], predictions)
            
            # Log model with signature
            if 'xgboost' in algorithm.lower():
                mlflow.xgboost.log_model(model, "model", signature=signature)
            elif 'lightgbm' in algorithm.lower():
                mlflow.lightgbm.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Copy tags from original run
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("algorithm", algorithm)
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("re_logged_with_signature", "true")
            mlflow.set_tag("original_run_id", original_run_id)
            
            # Copy metrics from original run
            metrics_to_copy = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for metric in metrics_to_copy:
                metric_col = f'metrics.{metric}'
                if metric_col in row and pd.notna(row[metric_col]):
                    mlflow.log_metric(metric, row[metric_col])
            
            # Copy parameters from original run
            param_columns = [col for col in row.index if col.startswith('params.')]
            for param_col in param_columns:
                if pd.notna(row[param_col]):
                    param_name = param_col.replace('params.', '')
                    mlflow.log_param(param_name, row[param_col])
            
            new_run_id = new_run.info.run_id
            new_run_ids[algorithm] = new_run_id
            
            print(f"  ✓ Model re-logged with signature")
            print(f"    New run ID: {new_run_id}")
            
    except Exception as e:
        print(f"  ✗ Error processing {algorithm}: {str(e)}")
        continue

print("\n" + "=" * 70)
print(f"✓ Re-logged {len(new_run_ids)} models with signatures")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

if new_run_ids:
    print("\n✅ SUCCESS - Models Fixed!")
    print("\nNew runs with signatures:")
    for algorithm, run_id in new_run_ids.items():
        print(f"  {algorithm:20s}: {run_id}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Go to notebook 04_model_evaluation.py")
    print("  2. Run the notebook - models should now register successfully")
    print("  3. The new runs are tagged with 'best_model' = 'true'")
    print("=" * 70)
else:
    print("\n⚠️  No models were fixed")
    print("Possible reasons:")
    print("  - Models already have signatures")
    print("  - No models with 'best_model' tag found")
    print("  - Check experiment name in configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Run Notebook 03 Again
# MAGIC
# MAGIC The training notebook (03_model_training.py) has been updated to include signatures.
# MAGIC If you prefer, you can simply re-run notebook 03 to train new models with signatures.
# MAGIC
# MAGIC **Note:** This utility notebook is a quick fix for existing models.

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    SIGNATURE FIX COMPLETE                              ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  All models have been re-logged with proper signatures                ║
║  You can now proceed with model registration                          ║
║                                                                        ║
║  The updated training notebooks (03) now automatically include        ║
║  signatures, so this issue won't occur for future training runs.      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)

print("✓ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    'experiment_name': '/Users/loan_approval_models',
    'model_registry_name': 'loan_approval_model',
    'champion_tag': 'champion'
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to MLflow

# COMMAND ----------

spark.sql(f"USE CATALOG amine_charot")

print(f"✓ Using catalog: amine_charot")

# COMMAND ----------

import mlflow

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
parent_dir = f"/Users/{user}"
experiment_path = f"{parent_dir}/testexperiment"

# Ensure the parent directory exists
dbutils.fs.mkdirs(parent_dir)

# Create experiment if it does not exist
experiment = mlflow.get_experiment_by_name(experiment_path)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_path)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_path)

display(f"✓ Connected to experiment: {experiment_path}")
display(f"  Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve All Runs

# COMMAND ----------

# Get all runs from the experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id])

print(f"✓ Retrieved {len(runs)} runs from MLflow")
print(f"\nDataFrame shape: {runs.shape}")

# Display first few runs
display(runs.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Best Models from Each Algorithm

# COMMAND ----------

# Filter runs with 'best_model' tag
best_models = runs[runs['tags.best_model'] == 'true']

# If no tagged best models, get best from each algorithm type
if len(best_models) == 0:
    print("No runs tagged as 'best_model'. Selecting best from each algorithm...")
    
    # Get unique algorithms
    algorithms = runs['tags.algorithm'].unique()
    
    best_models_list = []
    for algo in algorithms:
        if pd.notna(algo):
            algo_runs = runs[runs['tags.algorithm'] == algo]
            if len(algo_runs) > 0 and 'metrics.roc_auc' in algo_runs.columns:
                best_run = algo_runs.loc[algo_runs['metrics.roc_auc'].idxmax()]
                best_models_list.append(best_run)
    
    best_models = pd.DataFrame(best_models_list)

print(f"\n✓ Selected {len(best_models)} best models for comparison")
print("\nBest models by algorithm:")
display(best_models[['tags.algorithm', 'tags.model_type', 'metrics.roc_auc', 
                      'metrics.accuracy', 'metrics.f1_score']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comprehensive Model Comparison

# COMMAND ----------

# Create comparison dataframe
comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

comparison_data = []
for idx, row in best_models.iterrows():
    model_info = {
        'run_id': row['run_id'],
        'model_name': row['tags.algorithm'] if pd.notna(row.get('tags.algorithm')) else 'Unknown',
        'model_type': row['tags.model_type'] if pd.notna(row.get('tags.model_type')) else 'Unknown'
    }
    
    # Add metrics
    for metric in comparison_metrics:
        metric_col = f'metrics.{metric}'
        if metric_col in row and pd.notna(row[metric_col]):
            model_info[metric] = row[metric_col]
        else:
            model_info[metric] = 0.0
    
    comparison_data.append(model_info)

comparison_df = pd.DataFrame(comparison_data)

# Sort by ROC-AUC
comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

print("Model Performance Comparison:")
print("=" * 90)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Model Comparison

# COMMAND ----------

# Create visualization comparing all metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, metric in enumerate(comparison_metrics):
    ax = axes[idx]
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
    bars = ax.barh(comparison_df['model_name'], comparison_df[metric], color=colors)
    
    # Customize plot
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    ax.set_xlim(0, 1.05)

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Radar Chart Comparison

# COMMAND ----------

# Create radar chart for model comparison
from math import pi

def create_radar_chart(comparison_df, metrics):
    """
    Create radar chart comparing models
    """
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))
    
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

radar_fig = create_radar_chart(comparison_df, comparison_metrics)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Champion Model

# COMMAND ----------

# Select champion based on ROC-AUC score
champion_row = comparison_df.iloc[0]
champion_run_id = champion_row['run_id']
champion_model_name = champion_row['model_name']
champion_roc_auc = champion_row['roc_auc']

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║                        CHAMPION MODEL SELECTED                         ║")
print("╠════════════════════════════════════════════════════════════════════════╣")
print(f"║  Model: {champion_model_name:<58s}║")
print(f"║  Run ID: {champion_run_id:<56s}║")
print("║                                                                        ║")
print("║  Performance Metrics:                                                  ║")
for metric in comparison_metrics:
    metric_value = champion_row[metric]
    print(f"║    {metric.replace('_', ' ').title():<20s}: {metric_value:.4f}{' ' * 36}║")
print("╚════════════════════════════════════════════════════════════════════════╝")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Champion Model for Detailed Analysis

# COMMAND ----------

champion_model_uri = f"runs:/{champion_run_id}/model"
champion_model = mlflow.pyfunc.load_model(champion_model_uri)

print(f"✓ Champion model loaded: {champion_model_name}")
print(f"  Model type: {type(champion_model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detailed Performance Analysis

# COMMAND ----------

# Load test data
database_name = 'loan_workshop'
feature_table = 'customer_loan_features'
table_path = f"{database_name}.{feature_table}"

df = spark.table(table_path).toPandas()

# Prepare data (same preprocessing as training)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Test data loaded: {len(X_test)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix

# COMMAND ----------

# Generate predictions (these may be class labels or probabilities depending on your PyFunc)
y_pred = champion_model.predict(X_test)

# If y_pred are probabilities, convert to class labels as needed
if y_pred.ndim > 1 and y_pred.shape[1] > 1:
    # If output is probability for each class, take argmax
    y_pred_labels = y_pred.argmax(axis=1)
elif y_pred.ndim == 1 and ((y_pred > 0).all() and (y_pred < 1).all()):
    # If output is probability for positive class, threshold at 0.5
    y_pred_labels = (y_pred >= 0.5).astype(int)
else:
    # Assume output is already class labels
    y_pred_labels = y_pred

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=['Rejected (0)', 'Approved (1)'],
    yticklabels=['Rejected (0)', 'Approved (1)'],
    ax=ax, annot_kws={'size': 16}
)

ax.set_title(f'{champion_model_name} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Calculate confusion matrix metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN):  {tn:5d} - Correctly predicted rejections")
print(f"  False Positives (FP): {fp:5d} - Incorrectly predicted approvals")
print(f"  False Negatives (FN): {fn:5d} - Incorrectly predicted rejections")
print(f"  True Positives (TP):  {tp:5d} - Correctly predicted approvals")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC Curve and Precision-Recall Curve

# COMMAND ----------

# Get model predictions
y_pred = champion_model.predict(X_test)

# Try to infer probabilities
y_pred_proba = None
if hasattr(champion_model, "predict_proba"):
    try:
        y_pred_proba = champion_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_proba = None
elif y_pred.ndim > 1 and y_pred.shape[1] > 1:
    y_pred_proba = y_pred[:, 1]
elif y_pred.ndim == 1 and ((y_pred > 0).all() and (y_pred < 1).all()):
    y_pred_proba = y_pred

if y_pred_proba is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{champion_model_name} - ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    axes[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{champion_model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("Probabilities are not available from the model. ROC and PR curves cannot be plotted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])

print("Classification Report:")
print("=" * 60)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance Analysis (for tree-based models)

# COMMAND ----------

if hasattr(champion_model, 'feature_importances_'):
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': champion_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    top_n = 20
    plt.figure(figsize=(12, 10))
    plt.barh(range(top_n), feature_importance['importance'].head(top_n), color='steelblue')
    plt.yticks(range(top_n), feature_importance['feature'].head(top_n))
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'{champion_model_name} - Top {top_n} Feature Importance', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 Most Important Features:")
    print("=" * 60)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.4f}")
else:
    print("Feature importance not available for this model type")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction Distribution Analysis

# COMMAND ----------

# Try to infer probabilities from model output
y_pred_proba = None
if hasattr(champion_model, "predict_proba"):
    try:
        y_pred_proba = champion_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_proba = None
elif 'y_pred' in locals():
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_proba = y_pred[:, 1]
        elif y_pred.ndim == 1 and ((y_pred > 0).all() and (y_pred < 1).all()):
            y_pred_proba = y_pred

if y_pred_proba is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution of predicted probabilities
    axes[0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Actual Rejected', color='red')
    axes[0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Actual Approved', color='green')
    axes[0].set_xlabel('Predicted Probability of Approval', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Prediction confidence
    confidence = np.maximum(y_pred_proba, 1 - y_pred_proba)
    axes[1].hist(confidence, bins=50, color='purple', alpha=0.7)
    axes[1].set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(x=confidence.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {confidence.mean():.3f}')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nPrediction Confidence Statistics:")
    print(f"  Mean confidence: {confidence.mean():.4f}")
    print(f"  Median confidence: {np.median(confidence):.4f}")
    print(f"  Min confidence: {confidence.min():.4f}")
    print(f"  Max confidence: {confidence.max():.4f}")
else:
    print("Predicted probabilities are not available from the model. Cannot plot probability distributions or confidence statistics.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Champion Model to Model Registry

# COMMAND ----------

# Register the champion model
model_name = config['model_registry_name']

# Register model
model_version = mlflow.register_model(
    model_uri=champion_model_uri,
    name=model_name
)

print(f"✓ Model registered to Model Registry")
print(f"  Model name: {model_name}")
print(f"  Version: {model_version.version}")
print(f"  Source run ID: {champion_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Model Description and Tags

# COMMAND ----------

# Update model version with description and tags
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=f"""
    Champion loan approval model - {champion_model_name}
    
    Performance Metrics:
    - ROC-AUC: {champion_roc_auc:.4f}
    - Accuracy: {champion_row['accuracy']:.4f}
    - Precision: {champion_row['precision']:.4f}
    - Recall: {champion_row['recall']:.4f}
    - F1 Score: {champion_row['f1_score']:.4f}
    
    Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    Features: {len(X.columns)}
    Training Samples: {len(X_train)}
    Test Samples: {len(X_test)}
    """
)

# Add tags
client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="champion",
    value="true"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="algorithm",
    value=champion_model_name
)

print("✓ Model description and tags updated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transition Model to Production

# COMMAND ----------

# Assign an alias (e.g., "champion") to the model version
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version.version
)

print(f"✓ Model version {model_version.version} assigned alias 'champion'")
print(f"\n🎉 Champion model is now ready for deployment!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry Summary

# COMMAND ----------

# Example: Set your full model name
catalog_name = "amine_charot"
database_name = 'loan_workshop'
full_model_name = f"{catalog_name}.{database_name}.{model_name}"

# Get all versions of the model
all_versions = client.search_model_versions(f"name='{full_model_name}'")

print(f"Model Registry Status for '{full_model_name}':")
print("=" * 80)

for version in all_versions:
    print(f"\nVersion: {version.version}")
    print(f"  Stage: {version.current_stage}")
    print(f"  Status: {version.status}")
    print(f"  Run ID: {version.run_id}")
    print(f"  Created: {pd.Timestamp(version.creation_timestamp, unit='ms')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Summary

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    MODEL EVALUATION COMPLETE                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ✓ All models compared and evaluated                                  ║
║  ✓ Champion model selected                                            ║
║  ✓ Model registered to Model Registry                                 ║
║  ✓ Model transitioned to Production                                   ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  Model ready for deployment and serving!                               ║
╚════════════════════════════════════════════════════════════════════════╝
""")

print(f"\nChampion Model Details:")
print(f"  Name: {model_name}")
print(f"  Algorithm: {champion_model_name}")
print(f"  Version: {model_version.version}")
print(f"  Stage: Production")
print(f"  ROC-AUC: {champion_roc_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Proceed to **05_Model_Registry_and_Serving** to:
# MAGIC - Deploy the model for real-time serving
# MAGIC - Create inference pipelines
# MAGIC - Test model serving endpoints
# MAGIC - Monitor model performance in production
# MAGIC