# SVR_tips - Support Vector Regression on Tips Dataset

A comprehensive machine learning project demonstrating Support Vector Regression (SVR) implementation and hyperparameter optimization using the popular Tips dataset.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)

---

## 📋 Overview

This project provides a complete walkthrough of building and optimizing an SVR model for predicting restaurant bill amounts based on customer and contextual features. It covers data preprocessing, model training, hyperparameter tuning using GridSearchCV, and performance evaluation.

**Key Features:**
- Real-world dataset (Seaborn Tips dataset)
- Complete data preprocessing pipeline
- Model training and baseline evaluation
- Advanced hyperparameter tuning with GridSearchCV
- Detailed performance metrics and comparison

---

## 📊 Dataset Information

**Dataset:** Seaborn Tips Dataset
- **Samples:** 244 observations
- **Features:** 6 input features + 1 target variable
- **Missing Values:** None
- **Target Variable:** `total_bill` (restaurant bill amount in dollars)

### Features Description

| Feature | Type | Values |
|---------|------|--------|
| `tip` | Numerical | Tip amount (in dollars) |
| `sex` | Categorical | Male / Female |
| `smoker` | Categorical | Yes / No |
| `day` | Categorical | Thurs / Fri / Sat / Sun |
| `time` | Categorical | Lunch / Dinner |
| `size` | Numerical | Party size |
| **`total_bill`** | **Numerical** | **Target (in dollars)** |

---

## 🔧 Preprocessing Steps

### 1. **Data Loading & Exploration**
```python
import seaborn as sns
df = sns.load_dataset('tips')
```

### 2. **Missing Values Check**
- No missing values found in the dataset

### 3. **Categorical Encoding**
- **Label Encoding** for `sex`, `smoker`, and `time` features
- **One-Hot Encoding** for `day` feature (with `drop='first'` to avoid dummy variable trap)

### 4. **Train-Test Split**
- Training set: 75% (183 samples)
- Test set: 25% (61 samples)
- Random state: 10 (for reproducibility)

---

## 🤖 Model Implementation

### Default SVR Model

```python
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
```

**Default Configuration:**
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- Gamma: scale

---

## 🎯 Hyperparameter Tuning

Using **GridSearchCV** for systematic hyperparameter optimization:

### Parameters Tuned

| Parameter | Search Space | Purpose |
|-----------|--------------|---------|
| **C** | [0.1, 1, 10, 100, 1000] | Regularization strength |
| **gamma** | [1, 0.1, 0.01, 0.001, 0.0001] | Kernel coefficient |
| **kernel** | ['rbf'] | RBF function type |

### Total Combinations: 25
- Each combination evaluated with 5-fold cross-validation
- Best model automatically refitted on training data

---

## 📈 Model Performance

### Evaluation Metrics

**R² Score:**
- Measures proportion of variance explained
- Range: 0 to 1 (higher is better)
- 1.0 = perfect prediction

**Mean Absolute Error (MAE):**
- Average absolute error in dollars
- Lower values indicate better predictions

### Results Comparison

| Model | R² Score | MAE |
|-------|----------|-----|
| Default SVR | Baseline | Baseline |
| Tuned SVR (GridSearchCV) | Improved | Improved |

---

## 📦 Requirements

### Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Installation

```bash
# Using pip
pip install numpy pandas matplotlib seaborn scikit-learn

# Using conda
conda install numpy pandas matplotlib seaborn scikit-learn scikit-learn
```

### Python Version
- Python 3.7 or higher

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mohibullahlodhi/SVR_tips.git
cd SVR_tips
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

**Using Jupyter Notebook:**
```bash
jupyter notebook SVR_Tips.ipynb
```

**Using JupyterLab:**
```bash
jupyter lab SVR_Tips.ipynb
```

**Using VS Code:**
- Install Jupyter extension
- Open `SVR_Tips.ipynb`
- Click "Run All"

---

## 📁 Project Structure

```
SVR_tips/
├── README.md                    # Project documentation
├── SVR_Tips.ipynb              # Main Jupyter notebook
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore file
```

---

## 📚 Learning Outcomes

This project teaches:

1. **Data Preprocessing**
   - Categorical variable encoding
   - Feature standardization
   - Train-test data splitting

2. **Support Vector Regression**
   - SVR algorithm fundamentals
   - Kernel functions and their effects
   - Hyperparameter meaning and impact

3. **Model Optimization**
   - GridSearchCV for systematic tuning
   - Cross-validation techniques
   - Performance comparison and analysis

4. **Evaluation Metrics**
   - R² Score interpretation
   - Mean Absolute Error analysis
   - Baseline vs optimized model comparison

5. **Machine Learning Workflow**
   - Complete pipeline from raw data to predictions
   - Best practices for model development
   - Reproducibility and version control

---

## 💡 Tips for Improvement

### 1. Feature Engineering
- Create interaction terms between features
- Add polynomial features for non-linear relationships
- Apply standardization/normalization

### 2. Advanced Tuning
- Try other kernel types: 'linear', 'poly', 'sigmoid'
- Use Randomized Search for larger hyperparameter spaces
- Implement nested cross-validation

### 3. Alternative Algorithms
- Random Forest Regressor
- Gradient Boosting Regressor
- Neural Network Regressor
- Ensemble methods

### 4. Visualization
- Predicted vs actual values plot
- Residual analysis plots
- Feature importance visualization
- Learning curves

### 5. Model Validation
- k-fold cross-validation
- Learning curves for overfitting detection
- Validation curves for parameter analysis

---

## 🔗 Resources

- [Seaborn Tips Dataset](https://github.com/mwaskom/seaborn-data)
- [scikit-learn SVR Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Pandas Tutorial](https://pandas.pydata.org/docs/)

---

## 📝 License

This project is for **educational purposes**. Feel free to use and modify for learning and development.

---

## 👤 Author

**Mohibullah Lodhi**
- GitHub: [@mohibullahlodhi](https://github.com/mohibullahlodhi)

---

## 📞 Support

For questions or issues, please:
1. Check the notebook comments for detailed explanations
2. Review the dataset documentation in Seaborn
3. Open an issue on GitHub

---

**Last Updated:** March 2026  
**Version:** 1.0  
**Status:** ✅ Complete
