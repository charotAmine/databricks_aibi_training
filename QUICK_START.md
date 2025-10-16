# Quick Start Guide

Get started with the Databricks ML Workshop in under 10 minutes!

## 🚀 Quick Setup (5 Minutes)

### 1. Create a Cluster
```
Name: ML-Workshop
Runtime: ML Runtime 13.0+
Mode: Single Node (for learning)
```

### 2. Import Notebooks
Upload these 6 notebooks to your Databricks workspace:
- `01_data_generation.py`
- `02_eda_feature_engineering.py`
- `03_model_training.py`
- `04_model_evaluation.py`
- `05_model_serving.py`
- `06_automl.py`

### 3. Start Your Cluster
Click **Start** on your cluster and wait ~2-3 minutes for it to be ready.

---

## 📘 Notebook Execution Order

**⚠️ IMPORTANT:** Run notebooks in this exact order!

### ✅ Step 1: Data Generation (10 min)
**Notebook:** `01_data_generation.py`
- Generates 10,000 synthetic loan applications
- Creates Delta table: `loan_workshop.customer_loan_applications`
- **Run all cells from top to bottom**

### ✅ Step 2: EDA & Features (15 min)
**Notebook:** `02_eda_feature_engineering.py`
- Analyzes data patterns
- Creates engineered features
- Saves to: `loan_workshop.customer_loan_features`
- **Run all cells from top to bottom**

### ✅ Step 3: Model Training (20 min)
**Notebook:** `03_model_training.py`
- Trains 4 different algorithms
- Logs experiments to MLflow
- Tunes hyperparameters
- **Run all cells from top to bottom**
- **💡 Tip:** Open MLflow UI in another tab to watch experiments

### ✅ Step 4: Model Evaluation (10 min)
**Notebook:** `04_model_evaluation.py`
- Compares all models
- Selects best model
- Registers to Model Registry
- **Run all cells from top to bottom**

### ✅ Step 5: Model Serving (10 min)
**Notebook:** `05_model_serving.py`
- Loads model from registry
- Makes predictions
- Shows inference patterns
- **Run all cells from top to bottom**

### ✅ Step 6: AutoML (15 min)
**Notebook:** `06_automl.py`
- Demonstrates Databricks AutoML
- Compares with manual training
- **Run all cells from top to bottom**
- **⏱️ Note:** AutoML takes ~10 minutes to run

---

## 🎯 What You'll Learn

| Notebook | Key Concepts | Duration |
|----------|-------------|----------|
| 01 | Delta Lake, Data Generation | 10 min |
| 02 | EDA, Feature Engineering | 15 min |
| 03 | MLflow, Model Training, Hyperparameter Tuning | 20 min |
| 04 | Model Evaluation, Model Registry | 10 min |
| 05 | Model Serving, Inference | 10 min |
| 06 | AutoML, Comparison | 15 min |
| **Total** | | **~80 min** |

---

## 🔍 Quick Navigation

### Check Your Progress

After each notebook, verify:

**After Notebook 01:**
```sql
SELECT COUNT(*) FROM loan_workshop.customer_loan_applications
-- Should show: 10000
```

**After Notebook 02:**
```sql
SELECT COUNT(*) FROM loan_workshop.customer_loan_features  
-- Should show: 10000
```

**After Notebook 03:**
- Go to **Experiments** → Find `/Users/loan_approval_models`
- Should see multiple runs with different algorithms

**After Notebook 04:**
- Go to **Models** → Find `loan_approval_model`
- Should see version 1 in Production stage

**After Notebook 05:**
- Check Delta table: `loan_workshop.loan_predictions`
- Should see ~100 prediction records

**After Notebook 06:**
- Check MLflow experiments for AutoML runs
- Compare AutoML vs manual training results

---

## 💡 Pro Tips

### Tip 1: Open MLflow in Separate Tab
While running Notebook 03, open the MLflow UI in another browser tab:
- Click **Experiments** icon in left sidebar
- Navigate to `/Users/loan_approval_models`
- Watch experiments appear in real-time!

### Tip 2: Use "Run All" Carefully
- Each notebook has ~15-20 cells
- Some cells take longer than others
- Watch for errors before proceeding

### Tip 3: Save Your Work
- Databricks auto-saves notebooks
- But you can manually save with `Cmd+S` (Mac) or `Ctrl+S` (Windows)

### Tip 4: Cluster Management
- Stop cluster when not in use to save costs
- Clusters auto-terminate after 120 minutes of inactivity (default)

### Tip 5: Experiment with Code
- After completing all notebooks, go back and experiment!
- Try different hyperparameters
- Add new features
- Change model algorithms

---

## 🐛 Troubleshooting

### Problem: "Table not found" error
**Solution:** Make sure you ran previous notebooks in order

### Problem: MLflow experiment not showing
**Solution:** Refresh the Experiments page, check experiment name matches

### Problem: Cluster is slow
**Solution:** 
- Use a larger cluster type
- Or reduce dataset size in `01_data_generation.py` (change `num_samples`)

### Problem: Import error for a library
**Solution:** Ensure you selected **ML Runtime** (not standard runtime)

### Problem: AutoML timeout
**Solution:** Increase `timeout_minutes` in notebook 06 configuration

---

## 📊 Expected Results

After completing all notebooks:

### Data
- ✅ 10,000 loan applications generated
- ✅ 15+ features created
- ✅ ~50% approval rate

### Models
- ✅ 4 algorithms trained
- ✅ 20+ experiment runs
- ✅ Best model ROC-AUC: ~0.85-0.95

### MLOps
- ✅ All experiments tracked in MLflow
- ✅ Model registered in Model Registry
- ✅ Model in Production stage
- ✅ Predictions logged to Delta Lake

---

## 🎓 Next Steps After Workshop

### Beginner
1. Run through notebooks again
2. Experiment with parameters
3. Try different train/test splits
4. Add more visualizations

### Intermediate
1. Add new features
2. Try different algorithms (e.g., Neural Networks)
3. Implement cross-validation
4. Create custom metrics

### Advanced
1. Implement model monitoring
2. Set up automated retraining
3. Deploy model to REST endpoint
4. Integrate with Feature Store
5. Add A/B testing framework

---

## 📚 Learn More

### Key Documentation
- [Databricks ML Guide](https://docs.databricks.com/machine-learning/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [AutoML Guide](https://docs.databricks.com/applications/machine-learning/automl.html)

### Related Topics
- Feature Store
- Model Serving Endpoints  
- Drift Monitoring
- CI/CD for ML
- A/B Testing

---

## ✅ Completion Checklist

- [ ] Cluster created and started
- [ ] All 6 notebooks imported
- [ ] Notebook 01: Data generation completed
- [ ] Notebook 02: EDA completed
- [ ] Notebook 03: Model training completed
- [ ] Notebook 04: Model evaluation completed
- [ ] Notebook 05: Model serving completed
- [ ] Notebook 06: AutoML completed
- [ ] MLflow experiments reviewed
- [ ] Model Registry explored
- [ ] Understand end-to-end ML workflow

---

## 🎉 Congratulations!

You've completed the Databricks ML Workshop!

You now know how to:
- ✅ Build end-to-end ML pipelines
- ✅ Track experiments with MLflow
- ✅ Manage models in production
- ✅ Use Databricks AutoML
- ✅ Follow MLOps best practices

**Keep experimenting and building! 🚀**

---

## 📧 Need Help?

- Check the full [README.md](README.md) for detailed information
- Review the [WORKSHOP_GUIDE.md](WORKSHOP_GUIDE.md) for facilitator notes
- Consult Databricks documentation
- Ask your workshop instructor

---

*Happy Learning! 🎓*

