# Fraud Detection Model README

## Overview

This Jupyter Notebook (`Fraud_Detection.ipynb`) implements a machine learning model to predict fraudulent transactions for a financial company, as per the provided business case. The dataset (`Fraud.csv`) contains 6,362,620 rows and 10 columns, describing financial transactions with features like transaction type, amount, and balances. The model uses a Random Forest Classifier to detect fraud, addressing eight key questions: data cleaning, model description, variable selection, performance evaluation, key fraud predictors, their interpretability, prevention strategies, and effectiveness evaluation.

The code leverages the data dictionary (`Data Dictionary.txt`) to guide feature engineering and interpretation, ensuring alignment with the dataset’s context. The notebook is designed to run in an Anaconda environment with Python 3.8+ and required libraries.

## Dataset and Data Dictionary

- **Dataset**: `Fraud.csv`
  - **Location**: https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV
  - **Source**: Google Drive Link
  - **Description**: Contains 6,362,620 transactions with 10 columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- **Data Dictionary**: `Data Dictionary.txt`
  - **Location**: https://drive.google.com/uc?id=1VQ-HAm0oHbv0GmDKP2iqqFNc5aI91OLn&export=download
  - **Purpose**: Provides column descriptions, informing data cleaning, feature engineering, and interpretation. Key points:
    - `type`: Transaction types (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
    - `isFraud`: Indicates fraudulent transactions (1 = fraud, 0 = non-fraud).
    - `isFlaggedFraud`: Flags transfers &gt;200,000 as illegal attempts.
    - Merchant accounts (`nameDest` starting with 'M') lack balance data.

## Prerequisites

- **Anaconda**: Install Anaconda from https://www.anaconda.com/products/distribution.
- **Python**: Version 3.8 or higher.
- **Libraries**:

  ```bash
  conda install pandas numpy scikit-learn matplotlib seaborn statsmodels
  pip install statsmodels  # If conda installation fails
  ```
- **Files**:
  - Ensure `Fraud.csv` is at https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV
  - Ensure `Data Dictionary.txt` is at `C:\Users\mariy\OneDrive\Desktop\Data Dictionary.txt` for reference (not programmatically loaded).

## Setup Instructions

1. **Create a Conda Environment**:

   ```bash
   conda create -n fraud_detection python=3.8
   conda activate fraud_detection
   ```
2. **Install Libraries**:

   ```bash
   conda install pandas numpy scikit-learn matplotlib seaborn statsmodels
   pip install statsmodels
   ```
3. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```
4. **Open or Create Notebook**:
   - Navigate to `the location where you want to save your Jupyter file`.
   - Open `Fraud_Detection.ipynb` or create a new notebook and rename it.

## Notebook Structure

The notebook is structured to address the eight business case questions through Markdown and Code cells:

 1. **Introduction (Markdown)**: Outlines the business context and questions.
 2. **Data Dictionary (Markdown)**: Summarizes `Data Dictionary.txt` for reference.
 3. **Data Loading (Code)**: Loads `Fraud.csv` and cleans the `type` column.
 4. **Data Cleaning (Markdown + Code)**:
    - Checks for missing values (none expected, per data dictionary).
    - Caps outliers in numerical columns using the IQR method.
    - Assesses multi-collinearity with Variance Inflation Factor (VIF).
 5. **Model Development (Markdown + Code)**:
    - Uses a Random Forest Classifier for its robustness to imbalanced data.
    - Features engineered: `balanceDiffOrig`, `balanceDiffDest`, `isMerchant`.
    - One-hot encodes `type` column.
    - Splits data (80% train, 20% test), scales features, and trains the model.
 6. **Variable Selection (Markdown + Code)**: Selects features based on domain knowledge and feature importance.
 7. **Model Performance (Markdown + Code)**: Evaluates using confusion matrix, classification report, ROC-AUC score, and ROC curve.
 8. **Key Factors and Interpretation (Markdown)**: Identifies predictors like `amount`, `balanceDiffOrig`, `type_TRANSFER`, `type_CASH-OUT` and validates their relevance.
 9. **Prevention Strategies (Markdown)**: Proposes real-time monitoring, 2FA, and more.
10. **Effectiveness Evaluation (Markdown)**: Suggests fraud rate tracking, A/B testing, etc.
11. **Summary (Markdown)**: Recaps findings and recommendations.

## Running the Notebook

1. **Open Notebook**: Open `Fraud_Detection.ipynb` in Jupyter.
2. **Run All Cells**: Go to Cell &gt; Run All or run cells sequentially (Shift + Enter).
3. **Expected Outputs**:
   - Data loading: Confirms `type` values and columns.
   - Missing values: Shows no unexpected nulls.
   - Outliers: Lists counts for numerical columns.
   - VIF: Displays multi-collinearity metrics.
   - Model: Trains successfully, showing feature importance and performance metrics.
   - Plots: Feature importance bar plot and ROC curve.
4. **Troubleshooting**:
   - **FileNotFoundError**: Verify `Fraud.csv` path. Re-download if needed.
   - **KeyError for** `type`: Check column name case (`type` vs. `Type`). Run:

     ```python
     print(df.columns)
     print(df['type'].unique())
     ```

     Update `pd.get_dummies(df, columns=['type'], drop_first=True)` if needed.
   - **Memory Issues**: Use chunking or sampling:

     ```python
     df = pd.read_csv(file_path, chunksize=100000)
     df = pd.concat(df_chunks)
     # or
     df = pd.read_csv(file_path).sample(frac=0.1, random_state=42)
     ```
   - **Low Fraud Recall**: Add SMOTE before training:

     ```python
     from imblearn.over_sampling import SMOTE
     smote = SMOTE(random_state=42)
     X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
     ```

## Key Considerations

- **Dataset Size**: The dataset is large (471MB). Use chunking or sampling if memory issues occur.
- **Imbalanced Data**: Fraudulent transactions (`isFraud=1`) are rare, so the model prioritizes recall and ROC-AUC.
- **Data Dictionary**: Guides feature engineering (e.g., `isMerchant` from `nameDest`) and interpretation (e.g., `isFlaggedFraud` for thresholds).
- **Performance**: Monitor recall for fraud detection and adjust hyperparameters if needed:

  ```python
  rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
  ```

## Exporting Results

- Save as `.ipynb`: File &gt; Download as &gt; Notebook (.ipynb).
- Export as HTML:

  ```bash
  jupyter nbconvert --to html Fraud_Detection.ipynb
  ```

## 