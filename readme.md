# Telecom Customer Churn Prediction

This project provides a comprehensive workflow for predicting customer churn in the telecom sector using Python and Jupyter Notebook. The workflow is implemented in `main.ipynb` and covers the following steps:

## Workflow Overview

1. **Data Loading & Cleaning**
   - Loads the dataset (`Telco_customer_churn.csv`).
   - Drops irrelevant columns and handles missing values.
   - Encodes categorical features for modeling.

2. **Handling Class Imbalance**
   - Upsamples the minority class (churned customers) in the training set to address class imbalance.

3. **Model Training & Evaluation**
   - **Logistic Regression:**
     - Trains a logistic regression model with L2 regularization.
     - Evaluates using accuracy, precision, recall, F1-score, and ROC-AUC.
     - Saves the trained model as `logistic_regression_model.joblib`.
   - **Gaussian Naive Bayes:**
     - Trains a GaussianNB model on numerical features.
     - Evaluates and saves as `gaussian_nb_model.joblib`.
   - **XGBoost Classifier:**
     - Trains an XGBoost model, evaluates, and saves as `xgboost_model.joblib`.

4. **Model Comparison & Visualization**
   - Compares all models using key metrics.
   - Visualizes feature importances, precision/recall, class distributions, and correlation heatmaps.

5. **Model Loading & Integrity Check**
   - Loads saved models and confirms their performance matches the original evaluation, ensuring reproducibility.

## How to Use

1. Open `main.ipynb` in Jupyter or VS Code.
2. Run all cells to reproduce the analysis, model training, evaluation, and visualizations.
3. Saved models (`*.joblib`) can be loaded for inference or further analysis.

## Requirements
- Python 3.12+
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

## Data
- The main dataset is `Telco_customer_churn.csv` (and `.xlsx` version).

## Outputs
- Trained models: `logistic_regression_model.joblib`, `gaussian_nb_model.joblib`, `xgboost_model.joblib`
- Visualizations: Plots are generated in the notebook for EDA, feature importance, and model comparison.

---

## Updates
1. **Added New Feature Engineering:**
   - Added 'Monthly_Cost_Ratio' = MonthlyCharges / TotalCharges
   - Added 'Tenure_Cost_Ratio' = TotalCharges / Tenure Months
   - Both features are now included in all model training

2. **Code Improvements:**
   - Fixed column name references to use correct 'Tenure Months' instead of 'tenure'
   - Added proper handling of infinite values and NaN in calculated ratios
   - Added visualization of new features' relationship with churn
   - Updated feature importance analysis to include new calculated columns

3. **New Visualizations:**
   - Added boxplots showing distribution of both new ratios by churn status
   - Added correlation analysis of new features with churn
   - Updated feature importance plots to include new calculated columns

These updates enhance the model's ability to capture the relationship between customer spending patterns, tenure, and churn probability.

---

For details, see the step-by-step workflow and code in `main.ipynb`.
