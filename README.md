# Support Vector Regression (SVR) - Tips Dataset Analysis

## Project Overview

This project demonstrates the implementation and optimization of **Support Vector Regression (SVR)** using the popular **Tips dataset**. The notebook (`SVR_Tips.ipynb`) provides a comprehensive guide on building, training, and tuning an SVR model to predict restaurant bill amounts based on various customer and contextual features.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Features and Target Variable](#features-and-target-variable)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results and Performance](#results-and-performance)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)

## Dataset Description

The dataset used in this project is the **Tips dataset** from Seaborn, which contains information about restaurant tips collected from a restaurant over a period of time. This is a real-world dataset commonly used for regression analysis and exploratory data analysis (EDA).

**Dataset Size:** 244 observations (rows)

**Overview:**
- The dataset contains both numerical and categorical features
- No missing values were found in the dataset
- The target variable is `total_bill` (the total amount of the restaurant bill in dollars)

### Dataset Features:

| Feature | Type | Description |
|---------|------|-------------|
| `total_bill` | Numerical | Total bill amount in dollars (Target Variable) |
| `tip` | Numerical | Tip amount in dollars |
| `sex` | Categorical | Gender of the person paying the bill (Male/Female) |
| `smoker` | Categorical | Whether the party included a smoker (Yes/No) |
| `day` | Categorical | Day of the week (Thurs/Fri/Sat/Sun) |
| `time` | Categorical | Time of day (Lunch/Dinner) |
| `size` | Numerical | Number of people in the party |

## Features and Target Variable

### Input Features (X):
- `tip` - Tip amount given
- `sex` - Gender of the bill payer
- `smoker` - Smoking status
- `day` - Day of the week
- `time` - Meal time (Lunch/Dinner)
- `size` - Party size

### Target Variable (y):
- `total_bill` - The total restaurant bill amount to be predicted

## Data Preprocessing

The notebook implements several preprocessing steps to prepare the data for modeling:

### 1. **Data Exploration**
   - Loading the dataset using Seaborn
   - Checking dataset shape, data types, and null values
   - Analyzing categorical features using value counts

### 2. **Feature Encoding**
   - **Label Encoding:** Applied to convert categorical variables (`sex`, `smoker`, `time`) to numerical values
   - **Label Encoders Used:**
     - `le1` - Encodes the `sex` feature (Male=1, Female=0)
     - `le2` - Encodes the `smoker` feature (Yes=1, No=0)
     - `le3` - Encodes the `time` feature (Dinner=1, Lunch=0)

### 3. **One-Hot Encoding**
   - Applied to the `day` feature using `ColumnTransformer`
   - `drop='first'` parameter is used to avoid the dummy variable trap
   - Reduces multicollinearity issues

### 4. **Train-Test Split**
   - Data split into 75% training and 25% testing sets
   - Random state = 10 for reproducibility

## Model Implementation

### Basic Support Vector Regression

The first model uses the default SVR configuration:

```python
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
```

**Key Parameters (Default):**
- Kernel: RBF (Radial Basis Function)
- C: 1.0 (Regularization parameter)
- Gamma: 'scale' (Kernel coefficient)

### Model Performance (Default SVR):
- **R² Score:** Baseline performance metric
- **Mean Absolute Error (MAE):** Baseline error in dollars

## Hyperparameter Tuning

To improve model performance, **GridSearchCV** is used for hyperparameter optimization:

### Tuning Parameters:

| Parameter | Values Tested | Purpose |
|-----------|---------------|---------|
| `C` | [0.1, 1, 10, 100, 1000] | Regularization strength - controls the trade-off between fitting the training data and avoiding overfitting |
| `gamma` | [1, 0.1, 0.01, 0.001, 0.0001] | Kernel coefficient - defines how far the influence of a single training example reaches |
| `kernel` | ['rbf'] | Function type - RBF (Radial Basis Function) kernel is used |

### GridSearchCV Configuration:
- **Total combinations:** 25 (5 C values × 5 gamma values × 1 kernel)
- **Cross-validation:** Uses default 5-fold cross-validation
- **Best model:** Automatically refit with the best parameters

**Best Parameters Found:**
The notebook displays the optimal hyperparameters determined by GridSearchCV that maximize model performance.

## Results and Performance

### Model Comparison:

#### 1. Default SVR Model
- Uses default hyperparameters
- Provides baseline performance metrics

#### 2. Tuned SVR Model (GridSearchCV)
- Uses optimized hyperparameters
- Improved performance metrics compared to the default model

### Evaluation Metrics:

**R² Score (Coefficient of Determination):**
- Measures the proportion of variance explained by the model
- Range: 0 to 1 (higher is better)
- 1.0 indicates perfect prediction

**Mean Absolute Error (MAE):**
- Average absolute difference between predicted and actual values
- Measured in dollars
- Lower values indicate better performance

### Performance Comparison:
The tuned model typically shows:
- Higher R² Score compared to the default model
- Lower MAE (better predictions)
- Improved generalization to unseen data

## Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Installation
To install the required packages, run:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Use

### 1. **Prerequisites**
   - Install Python and the required libraries (see Requirements section)
   - Have Jupyter Notebook or JupyterLab installed

### 2. **Running the Notebook**
   
   **Option A - Using Jupyter Notebook:**
   ```bash
   jupyter notebook SVR_Tips.ipynb
   ```
   
   **Option B - Using JupyterLab:**
   ```bash
   jupyter lab SVR_Tips.ipynb
   ```
   
   **Option C - Using VS Code:**
   - Open the notebook directly in VS Code with Jupyter extension installed
   - Click "Run All" to execute all cells sequentially

### 3. **Step-by-Step Execution**
   - Run cells in order from top to bottom
   - Each cell builds upon the previous ones
   - Monitor output messages to ensure proper execution

### 4. **Interpreting Results**
   - Review the model metrics displayed in the final cells
   - Compare the tuned model performance with the default model
   - Use the best parameters for production deployment if needed

## Project Structure

```
SVM/
│
├── SVR_Tips.ipynb                 # Main notebook containing all analyses
├── README.md                       # This file - project documentation
│
└── (Optional) Supporting files:
    ├── requirements.txt            # List of dependencies
    └── data/                       # (If saving the tips dataset locally)
        └── tips.csv
```

## Key Insights and Learning Objectives

This project teaches:

1. **Data Preprocessing Techniques:**
   - Handling categorical variables through encoding
   - Proper data transformation for machine learning models

2. **Support Vector Regression Fundamentals:**
   - Understanding SVR algorithm basics
   - Kernel functions and their effects
   - The role of hyperparameters (C, gamma)

3. **Model Optimization:**
   - GridSearchCV for systematic hyperparameter tuning
   - Cross-validation for robust performance estimation
   - Comparing baseline and optimized models

4. **Evaluation Metrics:**
   - R² Score interpretation
   - Mean Absolute Error analysis
   - Model performance comparison

5. **Machine Learning Workflow:**
   - Data loading and exploration
   - Preprocessing and feature engineering
   - Model training and evaluation
   - Hyperparameter optimization

## Tips for Further Improvements

### 1. **Feature Engineering**
   - Create interaction features between existing features
   - Try polynomial features for non-linear relationships
   - Normalize or standardize numerical features

### 2. **Model Variations**
   - Try different kernel functions: 'linear', 'poly', 'sigmoid'
   - Experiment with other regression algorithms (Random Forest, Gradient Boosting)
   - Use ensemble methods combining multiple models

### 3. **Cross-Validation**
   - Implement k-fold cross-validation for more robust results
   - Use stratified cross-validation for better sampling

### 4. **Visualization**
   - Plot predicted vs actual values
   - Create residual plots to analyze model errors
   - Visualize feature importance

### 5. **Performance Tracking**
   - Compare multiple metrics (RMSE, MAPE, etc.)
   - Create learning curves to detect overfitting/underfitting
   - Use validation curves for parameter analysis

## Author Notes

This notebook provides a practical implementation of Support Vector Regression on a real-world dataset. The emphasis is on:
- Understanding the complete machine learning pipeline
- Hands-on hyperparameter tuning
- Practical model evaluation and comparison

The Tips dataset is small enough to train quickly yet complex enough to demonstrate important machine learning concepts.

## References

- **Dataset:** Seaborn Tips Dataset
- **Libraries Used:**
  - [scikit-learn Documentation](https://scikit-learn.org/)
  - [Pandas Documentation](https://pandas.pydata.org/)
  - [Seaborn Documentation](https://seaborn.pydata.org/)

## License

This project is for educational purposes. Feel free to use and modify for learning and development.

---

**Last Updated:** 2026
**Version:** 1.0
#   S V R _ t i p s  
 