# Inventory Management Dataset Analysis

## Overview

This repository contains a comprehensive regression analysis and **machine learning solution** for an inventory management dataset. The analysis explores various factors that influence inventory levels and implements **predictive models to forecast urgent restocks before they happen**, providing actionable insights for optimizing inventory management strategies.

## Dataset Description

The **Inventory Management Dataset** contains 10,000 records with 12 features related to product inventory management across different categories.

### Dataset Structure
- **Size**: 10,000 rows × 12 columns
- **Data Types**: 
  - Numerical: 10 features (7 float64, 3 int64)
  - Categorical: 2 features (object)
- **Target Variable**: `Urgent_Restock` - Binary indicator for urgent restocking needs (0/1)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Product_ID` | int64 | Unique identifier for each product |
| `Category` | object | Product category (Clothing, Electronics, Furniture, Food) |
| `Supplier_Rating` | int64 | Rating of the supplier (1-5 scale) |
| `Lead_Time` | float64 | Time between order placement and delivery |
| `Inventory_Level` | float64 | Current stock level of the product |
| `Urgent_Restock` | int64 | **Target Variable** - Binary indicator for urgent restocking needs (0/1) |
| `Seasonal_Demand` | float64 | Seasonal demand factor (0-1 scale) |
| `Warehouse_Capacity` | float64 | Available warehouse capacity |
| `Transportation_Cost` | float64 | Cost of transportation |
| `Product_Lifecycle_Stage` | object | Stage in product lifecycle (Introductory, Growth, Maturity, Decline) |
| `Returns_Percentage` | float64 | Percentage of product returns |
| `Demand_Variability` | float64 | Variability in product demand (0-1 scale) |

## Machine Learning Models

This analysis implements **four different machine learning algorithms** to predict urgent restocks:

### Models Implemented
1. **XGBoost** - Gradient boosting ensemble method for high-performance predictions
2. **Elastic Net** - Regularized linear regression combining L1 and L2 penalties
3. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
4. **Linear Regression** - Traditional statistical approach for baseline comparison

### Prediction Objective
The models use the **inventory_management_dataset(in).csv** data to predict the `Urgent_Restock` variable, enabling proactive inventory management by identifying products that will require urgent restocking before stockouts occur.

## Analysis Methodology

### 1. Data Exploration & Understanding
- **Dataset Overview**: Comprehensive examination of data structure, types, and basic statistics
- **Missing Value Analysis**: Identified missing values only in the `Product_Lifecycle_Stage` column
- **Duplicate Check**: Confirmed no duplicate records in the dataset

### 2. Univariate Analysis
- **Categorical Variables**: Analysis of distribution across product categories and lifecycle stages
- **Numerical Variables**: Statistical summaries and distribution analysis
- **Visualization**: Comprehensive plotting of all variables to understand individual distributions

### 3. Multivariate Analysis
- **Correlation Analysis**: Examination of relationships between variables
- **Scatter Plot Matrix**: Pairwise relationships using Seaborn and Plotly
- **Hue-based Analysis**: Grouped analysis by product lifecycle stage and category

### 4. Data Preprocessing
- **Missing Value Treatment**: Analysis showed missing values follow similar patterns to the overall dataset
- **Categorical Encoding**: 
  - One-hot encoding for `Category` column
  - Ordinal encoding for `Product_Lifecycle_Stage`
- **Feature Scaling**: Applied standardization to numerical features
- **Outlier Detection**: Z-score analysis for identifying potential outliers

### 5. Machine Learning Pipeline
- **Feature Selection**: Identification of most predictive features for urgent restock prediction
- **Model Training**: Implementation of XGBoost, Elastic Net, Naive Bayes, and Linear Regression
- **Model Evaluation**: Performance comparison using accuracy, precision, recall, and F1-score
- **Cross-Validation**: Robust model validation to ensure generalizability

## Key Findings

### Missing Value Analysis
- Missing values appear only in the `Product_Lifecycle_Stage` column
- The analysis revealed that missing values follow the same statistical pattern as the overall dataset
- Missing values are most commonly associated with the 'Electronics' category

### Distribution Insights
- **Inventory Level**: Shows statistically valid distributions suitable for outlier detection
- **Product Categories**: Balanced distribution across Clothing, Electronics, Furniture, and Food
- **Lifecycle Stages**: Products span across all lifecycle stages from Introductory to Decline

### Correlation Patterns
- Identified significant relationships between various inventory management factors
- Transportation costs and warehouse capacity show important correlations with inventory levels
- Seasonal demand patterns influence inventory requirements across different product categories

### Model Performance
- **Predictive Accuracy**: Comparative analysis of all four models for urgent restock prediction
- **Feature Importance**: Key factors driving urgent restock requirements
- **Business Impact**: Early warning system for inventory management optimization

## Technical Implementation

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
```

### Data Processing Pipeline
1. **Data Loading**: CSV file with 10,000 inventory records
2. **Exploratory Data Analysis**: Comprehensive statistical analysis
3. **Data Cleaning**: Missing value analysis and treatment
4. **Feature Engineering**: Categorical encoding and scaling
5. **Visualization**: Multi-dimensional plotting and analysis
6. **Model Training**: Implementation of four ML algorithms
7. **Model Evaluation**: Performance assessment and comparison

## Files in Repository

- `Dataset_1__Inventory_Management.ipynb` - Complete analysis notebook with ML models
- `inventory_management_dataset(in).csv` - Raw dataset (10,000 records)
- `README.md` - This documentation file

## Usage

To reproduce the analysis:

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy xgboost
   ```
3. Open and run the Jupyter notebook: `Dataset_1__Inventory_Management.ipynb`

## Analysis Highlights

### Predictive Modeling
This analysis implements **machine learning solutions** to predict urgent restocks before they happen, using four different algorithms to provide robust and reliable predictions for inventory management.

### Regression Context
The analysis combines traditional regression analysis for inventory level understanding with **classification models** for urgent restock prediction.

### Visualization Features
- Interactive Plotly visualizations for enhanced data exploration
- Comprehensive pairplot analysis using Seaborn
- Hue-based categorical analysis for pattern identification
- Model performance visualization and comparison

### Statistical Validation
- Z-score analysis for outlier identification
- Correlation matrix for feature relationship understanding
- Cross-validation for model reliability assessment
- Performance metrics for model comparison

## Business Value

- **Proactive Inventory Management**: Predict urgent restocks before stockouts occur
- **Cost Optimization**: Reduce emergency procurement costs and stockout penalties
- **Supply Chain Efficiency**: Improve planning and reduce inventory holding costs
- **Data-Driven Decisions**: Evidence-based approach to inventory optimization

## Future Work

- **Real-time Prediction System**: Deployment of models for live inventory monitoring
- **Time Series Analysis**: Seasonal demand pattern forecasting
- **Advanced Feature Engineering**: Development of derived features for improved accuracy
- **Model Ensemble**: Combining multiple models for enhanced prediction performance
- **Cost-Benefit Analysis**: ROI calculation for predictive inventory management
