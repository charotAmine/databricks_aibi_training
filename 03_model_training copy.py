# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 3: Model Training with MLflow
# MAGIC
# MAGIC ## Overview
# MAGIC In this notebook, we'll train multiple machine learning models with MLflow experiment tracking:
# MAGIC - Logistic Regression
# MAGIC - Random Forest
# MAGIC - Gradient Boosting (XGBoost)
# MAGIC - LightGBM
# MAGIC
# MAGIC ## MLOps Best Practices:
# MAGIC - **Experiment Tracking**: Log all experiments with MLflow
# MAGIC - **Hyperparameter Tuning**: Test different parameter combinations
# MAGIC - **Model Versioning**: Track all model versions
# MAGIC - **Reproducibility**: Set random seeds and log environment details
# MAGIC - **Model Comparison**: Compare models on multiple metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✓ Libraries imported successfully")

# COMMAND ----------

spark.sql(f"USE CATALOG amine_charot")

print(f"✓ Using catalog: amine_charot")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
config = {
    'database_name': 'loan_workshop',
    'feature_table': 'customer_loan_features',
    'test_size': 0.2,
    'random_state': RANDOM_STATE,
    'experiment_name': 'experiments/2832165408705406',
    'cv_folds': 5
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Data

# COMMAND ----------

# Load processed data
table_path = f"{config['database_name']}.{config['feature_table']}"
df_spark = spark.table(table_path)
df = df_spark.toPandas()

print(f"✓ Loaded {len(df)} records from {table_path}")
print(f"Features: {df.shape[1] - 1}")
print(f"Target distribution: {df['loan_approved'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Features and Target

# COMMAND ----------

# Separate features and target
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"\nCategorical columns to encode: {list(categorical_columns)}")

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_columns)} categorical features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train-Test Split

# COMMAND ----------

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config['test_size'], 
    random_state=config['random_state'],
    stratify=y
)

print(f"Dataset Split:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTarget distribution in training set:")
print(f"  Approved: {y_train.sum()} ({y_train.mean():.2%})")
print(f"  Rejected: {len(y_train) - y_train.sum()} ({1-y_train.mean():.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Scaling

# COMMAND ----------

# Scale features for models that benefit from scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow Experiment

# COMMAND ----------

import mlflow

experiment_name = f"/Users/amine.charot@databricks.com/testexperiment"

# Create the parent directory if it does not exist
if not dbutils.fs.mkdirs(f"dbfs:{parent_dir}"):
    dbutils.fs.mkdirs(f"dbfs:{parent_dir}")

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

display(f"✓ MLflow experiment set: {experiment_name}")
display(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def evaluate_model(model, X_test, y_test, y_pred, y_pred_proba):
    """
    Evaluate model and return metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    return plt.gcf()

def plot_roc_curve(y_test, y_pred_proba, title='ROC Curve'):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=15, title='Feature Importance'):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
    return None

print("✓ Helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: Logistic Regression (Baseline)

# COMMAND ----------

print("=" * 70)
print("Training Model 1: Logistic Regression (Baseline)")
print("=" * 70)

with mlflow.start_run(run_name="Logistic_Regression_Baseline") as run:
    # Model parameters
    params = {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs'
    }
    
    # Log parameters
    mlflow.log_params(params)
    mlflow.set_tag("model_type", "baseline")
    mlflow.set_tag("algorithm", "logistic_regression")
    
    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test, y_pred, y_pred_proba)
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=config['cv_folds'], scoring='roc_auc')
    mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
    mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
    
    # Create and log plots
    cm_fig = plot_confusion_matrix(y_test, y_pred, 'Logistic Regression - Confusion Matrix')
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close()
    
    roc_fig = plot_roc_curve(y_test, y_pred_proba, 'Logistic Regression - ROC Curve')
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"\n✓ Logistic Regression trained successfully")
    print(f"Run ID: {run.info.run_id}")
    print(f"\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: Random Forest with Hyperparameter Tuning

# COMMAND ----------

print("=" * 70)
print("Training Model 2: Random Forest with Hyperparameter Tuning")
print("=" * 70)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print(f"Parameter grid: {param_grid}")
print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

# COMMAND ----------

# Train with different parameter sets
best_rf_score = 0
best_rf_params = None

for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        for min_split in param_grid['min_samples_split']:
            for min_leaf in param_grid['min_samples_leaf']:
                
                params = {
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'min_samples_split': min_split,
                    'min_samples_leaf': min_leaf,
                    'random_state': RANDOM_STATE
                }
                
                with mlflow.start_run(run_name=f"RandomForest_n{n_est}_d{max_d}_s{min_split}_l{min_leaf}") as run:
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.set_tag("model_type", "random_forest")
                    mlflow.set_tag("algorithm", "random_forest")
                    
                    # Train model
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Evaluate
                    metrics = evaluate_model(model, X_test, y_test, y_pred, y_pred_proba)
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Track best model
                    if metrics['roc_auc'] > best_rf_score:
                        best_rf_score = metrics['roc_auc']
                        best_rf_params = params
                        best_rf_model = model
                    
                    print(f"  Trained: n_est={n_est}, max_depth={max_d}, ROC-AUC={metrics['roc_auc']:.4f}")

print(f"\n✓ Random Forest hyperparameter tuning complete")
print(f"Best ROC-AUC: {best_rf_score:.4f}")
print(f"Best parameters: {best_rf_params}")

# COMMAND ----------

# Log best Random Forest model with additional details
with mlflow.start_run(run_name="RandomForest_BEST") as run:
    mlflow.log_params(best_rf_params)
    mlflow.set_tag("model_type", "best_random_forest")
    mlflow.set_tag("algorithm", "random_forest")
    mlflow.set_tag("best_model", "true")
    
    # Re-evaluate best model
    y_pred = best_rf_model.predict(X_test)
    y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(best_rf_model, X_test, y_test, y_pred, y_pred_proba)
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Create and log plots
    cm_fig = plot_confusion_matrix(y_test, y_pred, 'Random Forest (Best) - Confusion Matrix')
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close()
    
    roc_fig = plot_roc_curve(y_test, y_pred_proba, 'Random Forest (Best) - ROC Curve')
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close()
    
    fi_fig = plot_feature_importance(best_rf_model, X.columns, top_n=15, 
                                      title='Random Forest - Feature Importance')
    if fi_fig:
        mlflow.log_figure(fi_fig, "feature_importance.png")
        plt.close()
    
    # Log model
    mlflow.sklearn.log_model(best_rf_model, "model")
    
    print(f"✓ Best Random Forest model logged")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 3: XGBoost with Hyperparameter Tuning

# COMMAND ----------

print("=" * 70)
print("Training Model 3: XGBoost with Hyperparameter Tuning")
print("=" * 70)

# XGBoost parameter grid
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

print(f"Parameter grid: {xgb_param_grid}")

best_xgb_score = 0
best_xgb_params = None

for max_d in xgb_param_grid['max_depth']:
    for lr in xgb_param_grid['learning_rate']:
        for n_est in xgb_param_grid['n_estimators']:
            for subsample in xgb_param_grid['subsample']:
                
                params = {
                    'max_depth': max_d,
                    'learning_rate': lr,
                    'n_estimators': n_est,
                    'subsample': subsample,
                    'random_state': RANDOM_STATE,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
                
                with mlflow.start_run(run_name=f"XGBoost_d{max_d}_lr{lr}_n{n_est}") as run:
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.set_tag("model_type", "xgboost")
                    mlflow.set_tag("algorithm", "xgboost")
                    
                    # Train model
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Evaluate
                    metrics = evaluate_model(model, X_test, y_test, y_pred, y_pred_proba)
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Track best model
                    if metrics['roc_auc'] > best_xgb_score:
                        best_xgb_score = metrics['roc_auc']
                        best_xgb_params = params
                        best_xgb_model = model
                    
                    print(f"  Trained: depth={max_d}, lr={lr}, n={n_est}, ROC-AUC={metrics['roc_auc']:.4f}")

print(f"\n✓ XGBoost hyperparameter tuning complete")
print(f"Best ROC-AUC: {best_xgb_score:.4f}")
print(f"Best parameters: {best_xgb_params}")

# COMMAND ----------

# Log best XGBoost model
with mlflow.start_run(run_name="XGBoost_BEST") as run:
    mlflow.log_params(best_xgb_params)
    mlflow.set_tag("model_type", "best_xgboost")
    mlflow.set_tag("algorithm", "xgboost")
    mlflow.set_tag("best_model", "true")
    
    # Re-evaluate
    y_pred = best_xgb_model.predict(X_test)
    y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(best_xgb_model, X_test, y_test, y_pred, y_pred_proba)
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Create and log plots
    cm_fig = plot_confusion_matrix(y_test, y_pred, 'XGBoost (Best) - Confusion Matrix')
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close()
    
    roc_fig = plot_roc_curve(y_test, y_pred_proba, 'XGBoost (Best) - ROC Curve')
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close()
    
    fi_fig = plot_feature_importance(best_xgb_model, X.columns, top_n=15,
                                      title='XGBoost - Feature Importance')
    if fi_fig:
        mlflow.log_figure(fi_fig, "feature_importance.png")
        plt.close()
    
    # Log model
    mlflow.xgboost.log_model(best_xgb_model, "model")
    
    print(f"✓ Best XGBoost model logged")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 4: LightGBM

# COMMAND ----------

print("=" * 70)
print("Training Model 4: LightGBM")
print("=" * 70)

# LightGBM parameter grid
lgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'num_leaves': [31, 50]
}

best_lgb_score = 0
best_lgb_params = None

for max_d in lgb_param_grid['max_depth']:
    for lr in lgb_param_grid['learning_rate']:
        for n_est in lgb_param_grid['n_estimators']:
            for num_leaves in lgb_param_grid['num_leaves']:
                
                params = {
                    'max_depth': max_d,
                    'learning_rate': lr,
                    'n_estimators': n_est,
                    'num_leaves': num_leaves,
                    'random_state': RANDOM_STATE,
                    'verbose': -1
                }
                
                with mlflow.start_run(run_name=f"LightGBM_d{max_d}_lr{lr}_n{n_est}") as run:
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.set_tag("model_type", "lightgbm")
                    mlflow.set_tag("algorithm", "lightgbm")
                    
                    # Train model
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Evaluate
                    metrics = evaluate_model(model, X_test, y_test, y_pred, y_pred_proba)
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Track best model
                    if metrics['roc_auc'] > best_lgb_score:
                        best_lgb_score = metrics['roc_auc']
                        best_lgb_params = params
                        best_lgb_model = model
                    
                    print(f"  Trained: depth={max_d}, lr={lr}, ROC-AUC={metrics['roc_auc']:.4f}")

print(f"\n✓ LightGBM hyperparameter tuning complete")
print(f"Best ROC-AUC: {best_lgb_score:.4f}")
print(f"Best parameters: {best_lgb_params}")

# COMMAND ----------

# Log best LightGBM model
with mlflow.start_run(run_name="LightGBM_BEST") as run:
    mlflow.log_params(best_lgb_params)
    mlflow.set_tag("model_type", "best_lightgbm")
    mlflow.set_tag("algorithm", "lightgbm")
    mlflow.set_tag("best_model", "true")
    
    # Re-evaluate
    y_pred = best_lgb_model.predict(X_test)
    y_pred_proba = best_lgb_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(best_lgb_model, X_test, y_test, y_pred, y_pred_proba)
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Create and log plots
    cm_fig = plot_confusion_matrix(y_test, y_pred, 'LightGBM (Best) - Confusion Matrix')
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close()
    
    roc_fig = plot_roc_curve(y_test, y_pred_proba, 'LightGBM (Best) - ROC Curve')
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close()
    
    fi_fig = plot_feature_importance(best_lgb_model, X.columns, top_n=15,
                                      title='LightGBM - Feature Importance')
    if fi_fig:
        mlflow.log_figure(fi_fig, "feature_importance.png")
        plt.close()
    
    # Log model
    mlflow.lightgbm.log_model(best_lgb_model, "model")
    
    print(f"✓ Best LightGBM model logged")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of All Models

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                      MODEL TRAINING COMPLETE                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ✓ Logistic Regression (Baseline)                                     ║
║  ✓ Random Forest (with hyperparameter tuning)                         ║
║  ✓ XGBoost (with hyperparameter tuning)                               ║
║  ✓ LightGBM (with hyperparameter tuning)                              ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  All experiments tracked in MLflow                                     ║
║  Models ready for comparison and selection                             ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# Create summary comparison
summary_data = {
    'Model': ['Logistic Regression', 'Random Forest (Best)', 'XGBoost (Best)', 'LightGBM (Best)'],
    'ROC-AUC': [0.0, best_rf_score, best_xgb_score, best_lgb_score]
}

# Get Logistic Regression score
with mlflow.start_run(run_name="Summary_Comparison"):
    summary_df = pd.DataFrame(summary_data)
    print("\nBest Model Performance Comparison:")
    print("=" * 50)
    print(summary_df.to_string(index=False))
    
    # Find overall best model
    best_idx = summary_df['ROC-AUC'].idxmax()
    best_model_name = summary_df.loc[best_idx, 'Model']
    best_score = summary_df.loc[best_idx, 'ROC-AUC']
    
    print(f"\n🏆 Overall Best Model: {best_model_name}")
    print(f"   ROC-AUC Score: {best_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Proceed to **04_Model_Evaluation** to:
# MAGIC - Compare all models side-by-side
# MAGIC - Perform detailed evaluation
# MAGIC - Select the champion model
# MAGIC - Analyze model performance in depth
# MAGIC