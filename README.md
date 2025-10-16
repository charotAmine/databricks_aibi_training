# Databricks ML Workshop: Customer Loan Decision

A comprehensive hands-on workshop demonstrating Databricks ML and MLOps capabilities using a customer loan approval use case.

## 🎯 Workshop Overview

This workshop is designed for **data scientists** and **ML engineers** who want to learn Databricks ML capabilities including:

- **Data Engineering**: Generate and prepare synthetic loan application data
- **Feature Engineering**: Create meaningful features for loan decisions
- **Model Training**: Train multiple ML models with MLflow tracking
- **Hyperparameter Tuning**: Optimize model performance systematically
- **Model Evaluation**: Compare and select the best performing model
- **MLOps**: Model registry, versioning, and deployment
- **Model Serving**: Real-time and batch inference
- **AutoML**: Automated machine learning with Databricks AutoML

## 📋 Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of machine learning concepts
- Familiarity with scikit-learn (helpful but not required)

### Databricks Environment
- Databricks workspace (Community Edition or Standard)
- Cluster with ML runtime (recommended: ML Runtime 13.0 or later)
- Python 3.8+

### Required Libraries
The following libraries should be available in Databricks ML Runtime:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `mlflow`
- `matplotlib`
- `seaborn`

## 📚 Workshop Structure

The workshop consists of 6 progressive notebooks:

### 1️⃣ **Data Generation** (`01_data_generation.py`)
- Generate synthetic customer loan application data
- Create realistic features (credit score, income, employment, etc.)
- Store data in Delta Lake
- Data quality validation

**Duration:** 15-20 minutes  
**Key Concepts:** Delta Lake, Data Generation, Feature Design

---

### 2️⃣ **EDA & Feature Engineering** (`02_eda_feature_engineering.py`)
- Exploratory data analysis with visualizations
- Identify patterns and correlations
- Engineer new features (financial health score, risk indicators, etc.)
- Prepare data for machine learning

**Duration:** 25-30 minutes  
**Key Concepts:** EDA, Feature Engineering, Data Visualization

---

### 3️⃣ **Model Training** (`03_model_training.py`)
- Train multiple classification models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
- Hyperparameter tuning for each algorithm
- MLflow experiment tracking
- Model versioning and logging

**Duration:** 30-40 minutes  
**Key Concepts:** MLflow, Experiment Tracking, Hyperparameter Tuning, Model Comparison

---

### 4️⃣ **Model Evaluation** (`04_model_evaluation.py`)
- Compare all trained models
- Detailed performance analysis (ROC curves, confusion matrices, etc.)
- Feature importance analysis
- Select champion model
- Register model to Model Registry
- Transition model to Production stage

**Duration:** 20-25 minutes  
**Key Concepts:** Model Evaluation, Model Registry, Model Selection

---

### 5️⃣ **Model Serving** (`05_model_serving.py`)
- Load model from Model Registry
- Batch scoring demonstrations
- Real-time prediction examples
- Model monitoring metrics
- Create inference API
- Save predictions to Delta Lake

**Duration:** 20-25 minutes  
**Key Concepts:** Model Serving, Inference Pipeline, Model Monitoring

---

### 6️⃣ **AutoML** (`06_automl.py`)
- Databricks AutoML demonstration
- Automatic model training and tuning
- Compare AutoML vs manual training
- Review generated notebooks
- Feature importance from AutoML

**Duration:** 20-30 minutes  
**Key Concepts:** AutoML, Automated ML, Model Comparison

---

## 🚀 Getting Started

### Step 1: Import Notebooks

1. Log in to your Databricks workspace
2. Navigate to your workspace home directory
3. Click **Import** in the top-right menu
4. Select **File** and upload all 6 notebook files (`.py` files)
5. Alternatively, you can import from this repository using Git integration

### Step 2: Create a Cluster

1. Go to **Compute** in the left sidebar
2. Click **Create Cluster**
3. Configure your cluster:
   - **Name:** `ML-Workshop-Cluster`
   - **Cluster Mode:** Single Node (for learning) or Standard (for larger datasets)
   - **Databricks Runtime Version:** ML Runtime 13.0+ (select ML version)
   - **Node Type:** Select based on your needs (e.g., `m5.large` for AWS)
4. Click **Create Cluster**

### Step 3: Run the Notebooks

**Important:** Run notebooks in order (01 → 06) as each builds on the previous one.

1. Open `01_data_generation.py`
2. Attach it to your cluster
3. Run all cells sequentially
4. Proceed to the next notebook after completion

### Step 4: Explore MLflow Experiments

1. Click on **Experiments** in the left sidebar
2. Locate the experiment: `/Users/loan_approval_models`
3. Explore different runs, compare metrics, and view artifacts

### Step 5: Check Model Registry

1. Click on **Models** in the left sidebar
2. Find the registered model: `loan_approval_model`
3. View different versions and stages

## 📊 Use Case: Customer Loan Decision

### Business Problem
A financial institution needs to automate loan approval decisions based on customer data to:
- Reduce manual review time
- Ensure consistent decision-making
- Minimize default risk
- Improve customer experience

### Dataset Features

#### Customer Demographics
- `age`: Customer age (18-75 years)
- `annual_income`: Annual income in USD
- `employment_length_years`: Years at current employment
- `employment_type`: Employment category (full_time, part_time, self_employed, contract)

#### Credit Information
- `credit_score`: Credit score (300-850)
- `existing_debt`: Current debt amount
- `debt_to_income_ratio`: Ratio of debt to income
- `num_credit_accounts`: Number of credit accounts
- `num_delinquencies`: Number of past delinquencies

#### Loan Details
- `loan_amount`: Requested loan amount
- `loan_purpose`: Purpose of loan (home_improvement, debt_consolidation, business, etc.)
- `loan_term_months`: Loan term in months

#### Engineered Features
- `loan_to_income_ratio`: Ratio of loan amount to annual income
- `estimated_monthly_payment`: Estimated monthly payment
- `payment_to_income_ratio`: Monthly payment as % of income
- `financial_health_score`: Composite financial health metric
- `risk_indicator`: Risk assessment score

#### Target Variable
- `loan_approved`: 1 = Approved, 0 = Rejected

## 🎓 Learning Objectives

By the end of this workshop, you will be able to:

1. ✅ **Generate and manage data** in Delta Lake
2. ✅ **Perform comprehensive EDA** and feature engineering
3. ✅ **Train multiple ML models** with different algorithms
4. ✅ **Track experiments** systematically with MLflow
5. ✅ **Tune hyperparameters** to optimize model performance
6. ✅ **Evaluate and compare models** using various metrics
7. ✅ **Register and version models** in Model Registry
8. ✅ **Deploy models** for real-time and batch inference
9. ✅ **Use Databricks AutoML** for rapid prototyping
10. ✅ **Implement MLOps best practices** in production

## 🔑 Key Technologies

| Technology | Purpose |
|-----------|---------|
| **Delta Lake** | Reliable data storage and ACID transactions |
| **MLflow** | Experiment tracking, model registry, and serving |
| **Databricks AutoML** | Automated machine learning |
| **scikit-learn** | Classical ML algorithms |
| **XGBoost** | Gradient boosting for structured data |
| **LightGBM** | Fast gradient boosting framework |
| **pandas** | Data manipulation and analysis |
| **matplotlib/seaborn** | Data visualization |

## 📈 Expected Results

After completing the workshop, you should see:

- **~10,000 loan applications** generated
- **15+ trained models** across different algorithms
- **Best model ROC-AUC** typically between 0.85-0.95
- **Multiple MLflow experiments** tracked and compared
- **Registered model** ready for production deployment
- **AutoML comparison** showing competitive performance

## 🎨 Workshop Highlights

### MLflow Experiment Tracking
- Automatic tracking of parameters, metrics, and artifacts
- Compare multiple runs side-by-side
- Visualize training metrics
- Store model artifacts and plots

### Model Registry
- Centralized model management
- Version control for models
- Stage transitions (Development → Staging → Production)
- Model lineage and metadata

### Databricks AutoML
- Automated feature engineering
- Automatic algorithm selection
- Hyperparameter optimization
- Generated notebooks for customization

### Real-World MLOps
- Reproducible pipelines
- Model monitoring concepts
- Batch and real-time scoring
- Production deployment patterns

## 🔧 Troubleshooting

### Common Issues

**Issue:** Model registration error - "signature metadata" required  
**Solution:** 
- If you see: "Model passed for registration did not contain any signature metadata"
- **Quick Fix:** Run the utility notebook `00_fix_signatures.py` to fix existing models
- **Or:** Re-run notebook 03 (training) - it now includes signatures automatically
- **Details:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for complete solution

**Issue:** Notebook fails with missing library  
**Solution:** Ensure you're using ML Runtime which includes all required libraries

**Issue:** Delta table not found  
**Solution:** Make sure to run notebooks in order (01 → 06)

**Issue:** MLflow experiment not showing  
**Solution:** Check experiment name in notebook configuration section

**Issue:** AutoML timeout  
**Solution:** Increase `timeout_minutes` in configuration (notebook 06)

**Issue:** Memory errors  
**Solution:** Use a larger cluster or reduce dataset size

**📖 For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

## 📖 Additional Resources

### Databricks Documentation
- [Databricks ML Guide](https://docs.databricks.com/machine-learning/index.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Delta Lake Guide](https://docs.databricks.com/delta/index.html)
- [AutoML Documentation](https://docs.databricks.com/applications/machine-learning/automl.html)

### Learning Paths
- [Databricks Academy](https://academy.databricks.com/)
- [ML Fundamentals](https://docs.databricks.com/machine-learning/train-model/index.html)
- [MLOps Best Practices](https://docs.databricks.com/machine-learning/mlops/mlops-workflow.html)

## 🤝 Contributing

This workshop is open for improvements! Feel free to:
- Report issues
- Suggest enhancements
- Add new features
- Improve documentation

## 📝 Workshop Feedback

We'd love to hear your feedback! After completing the workshop, please consider:
- What worked well?
- What could be improved?
- What additional topics would you like to see?
- How was the difficulty level?

## 📄 License

This workshop is provided as educational material for learning Databricks ML capabilities.

## 🙏 Acknowledgments

This workshop demonstrates:
- Databricks platform capabilities
- MLflow for experiment tracking
- Best practices in ML engineering
- Real-world MLOps patterns

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review Databricks documentation
3. Consult with your organization's Databricks administrator

---

## 🎉 Ready to Begin?

Start with **`01_data_generation.py`** and enjoy the workshop!

**Happy Learning! 🚀**

---

### Workshop Checklist

- [ ] Databricks workspace access
- [ ] Cluster created with ML Runtime
- [ ] All 6 notebooks imported
- [ ] Cluster started and ready
- [ ] Notebook 01 completed
- [ ] Notebook 02 completed
- [ ] Notebook 03 completed
- [ ] Notebook 04 completed
- [ ] Notebook 05 completed
- [ ] Notebook 06 completed
- [ ] Explored MLflow experiments
- [ ] Reviewed Model Registry
- [ ] Tested model predictions

**Total Workshop Duration:** ~2.5-3 hours

---

*Last Updated: October 2025*

