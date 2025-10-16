# Troubleshooting Guide

Common issues and solutions for the Databricks ML Workshop.

## 🔴 Model Signature Error (Unity Catalog)

### Error Message
```
MlflowException: Model passed for registration did not contain any signature metadata. 
All models in the Unity Catalog must be logged with a model signature containing both 
input and output type specifications.
```

### Why This Happens
Unity Catalog requires all models to have **signatures** that define:
- Input schema (feature names and types)
- Output schema (prediction types)

This ensures consistency and type safety when serving models.

### Solution Options

#### ✅ Option 1: Use the Fixed Notebooks (Recommended)
The training notebooks have been updated to automatically include signatures:

1. **Delete old MLflow runs** (optional but recommended):
   ```python
   # In Databricks notebook
   import mlflow
   from mlflow.tracking import MlflowClient
   
   client = MlflowClient()
   experiment = client.get_experiment_by_name('/Users/loan_approval_models')
   
   # Delete all runs in the experiment
   runs = client.search_runs(experiment.experiment_id)
   for run in runs:
       client.delete_run(run.info.run_id)
   ```

2. **Re-run notebook 03** (`03_model_training.py`):
   - The notebook now automatically adds signatures
   - All models will be logged with proper metadata

3. **Continue with notebook 04** (`04_model_evaluation.py`):
   - Model registration will now work correctly

---

#### ✅ Option 2: Use the Fix Utility (Quick Fix)
If you already trained models and don't want to re-train:

1. **Run the fix notebook** (`00_fix_signatures.py`):
   - This re-logs existing models with signatures
   - Preserves all metrics and parameters
   - Takes ~2-3 minutes

2. **Continue with notebook 04**:
   - Fixed models will register successfully

---

#### ✅ Option 3: Manual Fix (Advanced)
If you want to understand what's happening:

```python
import mlflow
from mlflow.models.signature import infer_signature

# Load your model
run_id = "YOUR_RUN_ID"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Load some sample data
X_sample = X_train[:100]  # Your feature data
y_sample = model.predict(X_sample)

# Infer signature
signature = infer_signature(X_sample, y_sample)

# Re-log with signature
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model, 
        "model", 
        signature=signature  # ← This is the key!
    )
```

---

### What Was Fixed

The notebooks now include:

**Added Import:**
```python
from mlflow.models.signature import infer_signature
```

**When Logging Models:**
```python
# OLD (would fail with Unity Catalog)
mlflow.sklearn.log_model(model, "model")

# NEW (works with Unity Catalog)
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature)
```

---

## 🟡 Other Common Issues

### Issue: Table Not Found

**Error:** `Table loan_workshop.customer_loan_applications not found`

**Solution:** Run notebooks in order:
1. First run `01_data_generation.py`
2. Then run `02_eda_feature_engineering.py`
3. Then continue with remaining notebooks

---

### Issue: Cluster Library Missing

**Error:** `ModuleNotFoundError: No module named 'xgboost'`

**Solution:** Use **ML Runtime** (not standard runtime):
1. Go to your cluster configuration
2. Under "Databricks Runtime Version"
3. Select a version with "ML" in the name (e.g., "13.0 ML")
4. Restart cluster

---

### Issue: MLflow Experiment Not Found

**Error:** `Experiment '/Users/loan_approval_models' not found`

**Solution:** The experiment is created automatically on first run:
1. Make sure you ran notebook 03 at least once
2. Check experiment name matches in config
3. Verify you have write permissions

---

### Issue: Memory Error During Training

**Error:** `OutOfMemoryError` or cluster crashes

**Solution:** 
1. **Option A:** Reduce dataset size:
   ```python
   # In 01_data_generation.py, change:
   config = {
       'num_samples': 5000,  # Reduced from 10000
       ...
   }
   ```

2. **Option B:** Use a larger cluster:
   - Increase node size (e.g., from `m5.large` to `m5.xlarge`)
   - Add more worker nodes

---

### Issue: AutoML Timeout

**Error:** AutoML completes but no results shown

**Solution:** Increase timeout in notebook 06:
```python
config = {
    'timeout_minutes': 20,  # Increased from 10
    'max_trials': 10,       # Reduced from 20
}
```

---

### Issue: Model Registry Access Denied

**Error:** `PERMISSION_DENIED: User does not have permission to register model`

**Solution:**
1. Check your workspace permissions
2. For Unity Catalog:
   - You need `CREATE MODEL` permission on the catalog
   - Ask your workspace admin to grant permissions
3. Alternative: Use Workspace Model Registry instead:
   ```python
   # Use workspace registry (not Unity Catalog)
   mlflow.set_registry_uri("databricks")
   ```

---

### Issue: Predictions Not Saving

**Error:** Can't write to Delta table

**Solution:**
1. Check database exists:
   ```python
   spark.sql("CREATE DATABASE IF NOT EXISTS loan_workshop")
   ```

2. Verify write permissions
3. Check if table is locked by another process

---

## 🔍 Debug Tips

### View MLflow Experiments
1. Click **Experiments** in left sidebar
2. Navigate to `/Users/loan_approval_models`
3. View all runs, metrics, and artifacts

### Check Model Registry
1. Click **Models** in left sidebar
2. Find `loan_approval_model`
3. View versions and stages

### Verify Delta Tables
```python
# List tables
spark.sql("SHOW TABLES IN loan_workshop").show()

# Check table contents
spark.sql("SELECT COUNT(*) FROM loan_workshop.customer_loan_applications").show()

# View table schema
spark.sql("DESCRIBE loan_workshop.customer_loan_applications").show()
```

### Check Cluster Logs
1. Go to **Compute** → Your cluster
2. Click **Event Log** tab
3. Look for errors or warnings

---

## 📧 Still Having Issues?

1. **Check the README.md** for setup instructions
2. **Review the WORKSHOP_GUIDE.md** for facilitator notes
3. **Consult Databricks documentation:**
   - [MLflow Guide](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
   - [Unity Catalog Models](https://docs.databricks.com/machine-learning/manage-model-lifecycle/models-in-uc.html)
4. **Ask your workshop instructor**

---

## ✅ Quick Checklist

When encountering issues:

- [ ] Running ML Runtime (not standard)?
- [ ] Cluster started and healthy?
- [ ] Notebooks run in order (01→06)?
- [ ] Experiment name matches config?
- [ ] Database `loan_workshop` exists?
- [ ] Tables created successfully?
- [ ] Enough cluster resources?
- [ ] Workspace permissions correct?

---

## 🎯 Best Practices to Avoid Issues

1. **Always use ML Runtime** for ML workloads
2. **Run notebooks sequentially** from 01 to 06
3. **Check cell output** before proceeding
4. **Monitor cluster resources** during training
5. **Use consistent naming** for experiments and models
6. **Test with small data first** before scaling up
7. **Keep notebooks updated** to latest versions

---

*Last Updated: October 2025*

