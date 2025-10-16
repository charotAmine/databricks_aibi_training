# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Loan Decision Workshop - Part 6: AutoML
# MAGIC
# MAGIC ## Overview
# MAGIC In this notebook, we'll demonstrate **Databricks AutoML** capabilities:
# MAGIC - Automatic feature engineering
# MAGIC - Automatic model selection and training
# MAGIC - Automatic hyperparameter tuning
# MAGIC - Compare AutoML results with manually trained models
# MAGIC
# MAGIC ## Benefits of AutoML:
# MAGIC - **Speed**: Quickly iterate through multiple algorithms
# MAGIC - **Automation**: Automatically handle preprocessing and tuning
# MAGIC - **Insights**: Get detailed model explanations and feature importance
# MAGIC - **Best Practices**: Built-in MLOps best practices with MLflow
# MAGIC - **Transparency**: Full code generation for reproducibility

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is Databricks AutoML?
# MAGIC
# MAGIC Databricks AutoML is a managed service that:
# MAGIC 1. **Automatically prepares data** for machine learning
# MAGIC 2. **Trains multiple models** with different algorithms
# MAGIC 3. **Performs hyperparameter tuning** to optimize performance
# MAGIC 4. **Generates notebooks** with full code for transparency
# MAGIC 5. **Tracks experiments** in MLflow automatically
# MAGIC 6. **Provides model explanations** and feature importance
# MAGIC
# MAGIC ### Supported Problem Types:
# MAGIC - Classification (Binary & Multi-class)
# MAGIC - Regression
# MAGIC - Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from databricks import automl
import mlflow
from datetime import datetime

print("✓ Libraries imported successfully")
print(f"✓ AutoML version: {automl.__version__ if hasattr(automl, '__version__') else 'Latest'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    'database_name': 'loan_workshop',
    'feature_table': 'customer_loan_features',
    'target_column': 'loan_approved',
    'timeout_minutes': 10,  # Maximum time for AutoML to run
    'max_trials': 20,  # Maximum number of models to try
}

print("AutoML Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load data from Delta Lake
table_path = f"{config['database_name']}.{config['feature_table']}"
df = spark.table(table_path)

print(f"✓ Loaded data from: {table_path}")
print(f"  Total records: {df.count()}")
print(f"  Total features: {len(df.columns)}")

# Show sample
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation for AutoML

# COMMAND ----------

# AutoML works best with Spark DataFrames
# Ensure target column is present
target_col = config['target_column']

print(f"Target column: {target_col}")
print(f"\nTarget distribution:")
df.groupBy(target_col).count().show()

# Check for any data quality issues
print(f"\nData Quality Check:")
print(f"  Total rows: {df.count()}")
print(f"  Total columns: {len(df.columns)}")

# Show schema
print(f"\nSchema:")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run AutoML Classification
# MAGIC
# MAGIC This will:
# MAGIC 1. Automatically split data into train/validation/test sets
# MAGIC 2. Try multiple classification algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, etc.)
# MAGIC 3. Perform hyperparameter tuning for each algorithm
# MAGIC 4. Track all experiments in MLflow
# MAGIC 5. Generate notebooks with full code
# MAGIC 6. Select the best model based on validation metrics

# COMMAND ----------

print("=" * 70)
print("STARTING DATABRICKS AUTOML")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: {table_path}")
print(f"Target: {target_col}")
print(f"Timeout: {config['timeout_minutes']} minutes")
print(f"Max trials: {config['max_trials']}")
print("\nAutoML is now running...")
print("This may take several minutes. AutoML will:")
print("  1. Preprocess the data")
print("  2. Train multiple models")
print("  3. Tune hyperparameters")
print("  4. Evaluate and compare models")
print("  5. Generate detailed reports")
print("\n" + "=" * 70)

# COMMAND ----------

# Run AutoML
# Note: This is the actual AutoML run - it will take several minutes
summary = automl.classify(
    dataset=df,
    target_col=target_col,
    timeout_minutes=config['timeout_minutes'],
    max_trials=config['max_trials']
)

print("\n✓ AutoML run completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Results Summary

# COMMAND ----------

print("=" * 70)
print("AUTOML RESULTS SUMMARY")
print("=" * 70)

# Get experiment ID
experiment_id = summary.experiment.experiment_id
print(f"\nExperiment ID: {experiment_id}")
print(f"Experiment URL: {summary.experiment.name}")

# Best trial information
print(f"\n🏆 Best Trial:")
print(f"   Trial ID: {summary.best_trial.mlflow_run_id}")
print(f"   Model type: {summary.best_trial.model_description}")

# Best metrics
print(f"\n📊 Best Model Metrics:")
for metric_name, metric_value in summary.best_trial.metrics.items():
    print(f"   {metric_name}: {metric_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Best Model from AutoML

# COMMAND ----------

# Load the best model
best_model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
best_automl_model = mlflow.sklearn.load_model(best_model_uri)

print(f"✓ Best AutoML model loaded")
print(f"  Model type: {type(best_automl_model).__name__}")
print(f"  Run ID: {summary.best_trial.mlflow_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare AutoML with Manual Training

# COMMAND ----------

print("=" * 70)
print("COMPARISON: AutoML vs Manual Training")
print("=" * 70)

# Get manual training results (from previous notebooks)
manual_experiment_name = '/Users/loan_approval_models'

try:
    manual_runs = mlflow.search_runs(
        experiment_names=[manual_experiment_name],
        filter_string="tags.best_model = 'true'",
        order_by=["metrics.roc_auc DESC"]
    )
    
    if len(manual_runs) > 0:
        best_manual = manual_runs.iloc[0]
        
        print("\n📊 Performance Comparison:")
        print("\n" + "-" * 70)
        print(f"{'Metric':<20} {'AutoML':<20} {'Manual Training':<20} {'Winner':<10}")
        print("-" * 70)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        automl_wins = 0
        manual_wins = 0
        
        for metric in metrics_to_compare:
            automl_val = summary.best_trial.metrics.get(f'val_{metric}', 
                         summary.best_trial.metrics.get(metric, 0))
            manual_val = best_manual.get(f'metrics.{metric}', 0)
            
            winner = 'AutoML' if automl_val > manual_val else 'Manual'
            if automl_val > manual_val:
                automl_wins += 1
            else:
                manual_wins += 1
            
            symbol = '🏆' if winner == 'AutoML' else '  '
            print(f"{metric:<20} {automl_val:<20.4f} {manual_val:<20.4f} {symbol}{winner:<10}")
        
        print("-" * 70)
        print(f"\nOverall Winner: {'AutoML' if automl_wins > manual_wins else 'Manual Training'}")
        print(f"  AutoML: {automl_wins} metrics won")
        print(f"  Manual: {manual_wins} metrics won")
        
    else:
        print("\n⚠️  No manual training results found for comparison")
        print("   Please run notebooks 03 and 04 first to enable comparison")
        
except Exception as e:
    print(f"\n⚠️  Could not load manual training results: {str(e)}")
    print("   Showing AutoML results only")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Generated Notebooks

# COMMAND ----------

print("=" * 70)
print("AUTOML GENERATED ARTIFACTS")
print("=" * 70)

print(f"\n📓 Generated Notebooks:")
print(f"   AutoML has generated notebooks with full code for:")
print(f"   1. Data exploration")
print(f"   2. Best model training")
print(f"   3. All trial models")
print(f"\n   Check the Experiments UI to access these notebooks")

print(f"\n📊 Experiment Tracking:")
print(f"   All AutoML runs are tracked in MLflow")
print(f"   Experiment: {summary.experiment.name}")
print(f"   Total trials: {config['max_trials']}")

print(f"\n🎯 Best Model:")
print(f"   The best model has been automatically selected")
print(f"   based on validation metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance from AutoML

# COMMAND ----------

# Try to get feature importance if available
if hasattr(best_automl_model, 'feature_importances_'):
    import matplotlib.pyplot as plt
    
    # Get feature names
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_automl_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    top_n = min(15, len(feature_importance_df))
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), feature_importance_df['importance'].head(top_n), color='steelblue')
    plt.yticks(range(top_n), feature_importance_df['feature'].head(top_n))
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'AutoML Model - Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 Most Important Features (AutoML):")
    print("=" * 60)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.4f}")
else:
    print("Feature importance not available for this model type")
    print("Check the AutoML generated notebooks for detailed feature analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation on Test Set

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Prepare test data
df_pd = df.toPandas()
X = df_pd.drop(target_col, axis=1)
y = df_pd[target_col]

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data (same as manual training for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set size: {len(X_test)} samples")

# Make predictions
y_pred = best_automl_model.predict(X_test)
y_pred_proba = best_automl_model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

print("\n📊 AutoML Model Test Set Performance:")
print("=" * 50)
for metric, value in test_metrics.items():
    print(f"  {metric.replace('_', ' ').title():<20s}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

import seaborn as sns

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Rejected (0)', 'Approved (1)'],
            yticklabels=['Rejected (0)', 'Approved (1)'],
            annot_kws={'size': 16})
plt.title('AutoML Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ROC Curve

# COMMAND ----------

from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('AutoML Model - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights: AutoML vs Manual Training

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    KEY INSIGHTS: AutoML vs Manual                      ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ✅ ADVANTAGES OF AutoML:                                              ║
║                                                                        ║
║  1. SPEED & EFFICIENCY                                                 ║
║     • Trains multiple models automatically                             ║
║     • Saves data scientist time                                        ║
║     • Quick baseline for new problems                                  ║
║                                                                        ║
║  2. AUTOMATION                                                         ║
║     • Automatic feature preprocessing                                  ║
║     • Automatic hyperparameter tuning                                  ║
║     • Built-in cross-validation                                        ║
║                                                                        ║
║  3. BEST PRACTICES                                                     ║
║     • Follows MLOps best practices                                     ║
║     • Automatic experiment tracking                                    ║
║     • Reproducible results                                             ║
║                                                                        ║
║  4. TRANSPARENCY                                                       ║
║     • Generates full code notebooks                                    ║
║     • Can customize generated code                                     ║
║     • Complete model explainability                                    ║
║                                                                        ║
║  ✅ ADVANTAGES OF Manual Training:                                     ║
║                                                                        ║
║  1. CONTROL & CUSTOMIZATION                                            ║
║     • Full control over feature engineering                            ║
║     • Custom model architectures                                       ║
║     • Domain-specific optimizations                                    ║
║                                                                        ║
║  2. LEARNING & INSIGHTS                                                ║
║     • Deep understanding of data                                       ║
║     • Better feature interpretation                                    ║
║     • Domain knowledge integration                                     ║
║                                                                        ║
║  3. OPTIMIZATION                                                       ║
║     • Fine-tuned for specific metrics                                  ║
║     • Business-specific constraints                                    ║
║     • Advanced techniques                                              ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  💡 RECOMMENDATION:                                                    ║
║                                                                        ║
║  Use BOTH approaches:                                                  ║
║  1. Start with AutoML for quick baseline                              ║
║  2. Use manual training for optimization                              ║
║  3. Combine insights from both methods                                ║
║  4. Choose best model for production                                  ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register AutoML Model (Optional)

# COMMAND ----------

# You can register the AutoML model if it performs better
# Uncomment to register:

# model_name = "loan_approval_model_automl"
# registered_model = mlflow.register_model(
#     model_uri=best_model_uri,
#     name=model_name
# )
# 
# print(f"✓ AutoML model registered as: {model_name}")
# print(f"  Version: {registered_model.version}")

print("To register the AutoML model, uncomment the code above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                        AUTOML DEMONSTRATION COMPLETE                   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ✓ AutoML successfully trained multiple models                        ║
║  ✓ Best model automatically selected                                  ║
║  ✓ Model performance evaluated on test set                            ║
║  ✓ Compared with manually trained models                              ║
║  ✓ Feature importance analyzed                                        ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  Key Takeaways:                                                        ║
║                                                                        ║
║  • AutoML provides quick, high-quality baselines                      ║
║  • Generated notebooks enable customization                           ║
║  • Full integration with MLflow for tracking                          ║
║  • Can be combined with manual ML for best results                    ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                    WORKSHOP COMPLETE! 🎉                               ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC **Congratulations!** You've completed the Databricks ML Workshop!
# MAGIC
# MAGIC ### What You've Learned:
# MAGIC 1. ✅ Data generation and preparation
# MAGIC 2. ✅ Exploratory data analysis and feature engineering
# MAGIC 3. ✅ Model training with multiple algorithms
# MAGIC 4. ✅ MLflow experiment tracking
# MAGIC 5. ✅ Model evaluation and comparison
# MAGIC 6. ✅ Model registry and versioning
# MAGIC 7. ✅ Model serving and deployment
# MAGIC 8. ✅ Databricks AutoML
# MAGIC
# MAGIC ### To Continue Your Journey:
# MAGIC - Experiment with different features
# MAGIC - Try advanced algorithms
# MAGIC - Implement A/B testing
# MAGIC - Set up model monitoring
# MAGIC - Deploy to production endpoints
# MAGIC - Integrate with business applications

