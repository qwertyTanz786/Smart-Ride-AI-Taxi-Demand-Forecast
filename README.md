# Smart Mobility Demand Prediction 🚖

## Project Overview

This project predicts **Taxi Demand** using machine learning.

The goal is to help ride-hailing companies, taxi operators, and city planners anticipate transportation demand based on:

* Location
* Weather
* Events
* Vehicle type
* Customer behavior
* Temporal features

The model is trained using the **Smart Mobility Dataset** and uses **XGBoost Regressor** to forecast taxi demand.

---

# Business Problem

Predicting taxi demand accurately can help:

* Reduce passenger waiting times
* Improve fleet allocation
* Optimize driver deployment
* Improve traffic management
* Support smart city initiatives

---

# Dataset Information

Dataset: `Smart_Mobility_1M_Cleaned.csv`

Target Variable:

```text
Taxi_Demand
```

Dataset Size:

```text
1,000,000+ Records
```

---

# Feature Engineering

## Date Processing

The Date column is converted into datetime format:

```python
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
```

Extracted Features:

* Day
* WeekOfYear

```python
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
```

Removed:

```python
Date
Time
```

---

# Data Leakage Prevention

The following columns were removed because they reveal information about demand after it occurs.

```python
Ride_Request_Count
Surge_Multiplier
Dynamic_Pricing_Activated
Cancelled_Rides
```

These features would artificially inflate model performance and would not be available during real-world prediction.

---

# Categorical Encoding

One-Hot Encoding was applied to:

* Pickup_Zone
* Dropoff_Zone
* Road_Name
* Weather
* Event_Type
* Vehicle_Type
* Payment_Type
* Trip_Type
* Day_Type
* Customer_Loyalty_Level

```python
pd.get_dummies(..., drop_first=True)
```

---

# Model

Algorithm Used:

```text
XGBoost Regressor
```

Why XGBoost?

* Handles large datasets efficiently
* Captures non-linear relationships
* Handles mixed feature types
* State-of-the-art performance on tabular data

---

# Train-Test Split

```python
80% Training Data
20% Testing Data
```

```python
random_state = 42
```

---

# Evaluation Metrics

Metric Used:

```text
R² Score
```

Formula:

R² measures how much variance in Taxi Demand is explained by the model.

```text
1.0 = Perfect Model
0.0 = Same as predicting mean
<0 = Worse than predicting mean
```

---

# Results

## Model Performance

Prediction: 258.4009
R2 Score: 0.941375732421875
Train Score: 0.9436690807342529
Test Score: 0.941375732421875
Predicted: 258.4009
Actual: 257

# Project Workflow

```text
Raw Dataset
    ↓
Date Feature Engineering
    ↓
Leakage Removal
    ↓
One-Hot Encoding
    ↓
Train-Test Split
    ↓
XGBoost Training
    ↓
Prediction
    ↓
Evaluation
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost

---

# Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Feature importance analysis
* SHAP Explainability
* Demand forecasting dashboard
* Real-time prediction API using FastAPI
* Docker deployment
* MLflow experiment tracking
* Airflow pipeline automation

---

# Author

Tanishq Panchal

Machine Learning Engineer Portfolio Project

Smart Mobility Demand Prediction
