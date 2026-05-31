import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,classification_report
from xgboost import XGBRegressor
df = pd.read_csv(r"C:\Portfolio-ML\Datasets\Dubai_Smart_Mobility_1M_Cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df.drop(['Date', 'Time'], axis=1, inplace=True)
leakage_cols = [
    'Ride_Request_Count',
    'Surge_Multiplier',
    'Dynamic_Pricing_Activated',
    'Cancelled_Rides'
]
df.drop(columns=leakage_cols, inplace=True)
categorical_cols = [
    'Pickup_Zone',
    'Dropoff_Zone',
    'Road_Name',
    'Weather',
    'Event_Type',
    'Vehicle_Type',
    'Payment_Type',
    'Trip_Type',
    'Day_Type',
    'Customer_Loyalty_Level'
]
df = pd.get_dummies(df,columns=categorical_cols,drop_first=True)
X = df.drop('Taxi_Demand', axis=1)
Y = df['Taxi_Demand']
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
m = XGBRegressor()
m.fit(X_train, Y_train)
# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = m.predict(X_test)
print("Prediction:", predictions[0])
print("R2 Score:", r2_score(Y_test, predictions))
print("Train Score:", m.score(X_train, Y_train))
print("Test Score:", m.score(X_test, Y_test))
print("Predicted:", predictions[0])
print("Actual:", Y_test.iloc[0])