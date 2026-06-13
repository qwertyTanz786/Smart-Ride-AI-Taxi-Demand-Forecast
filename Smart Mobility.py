# ============================================
# SMART RIDE AI - Taxi Demand Forecasting
# + Demand Category
# + Surge Pricing Recommendation
# + Fleet Allocation
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
# Load Dataset
df = pd.read_csv(
    r"C:\Portfolio-ML\Datasets\Dubai_Smart_Mobility_1M_Cleaned.csv"
)
# --------------------------------------------
# Feature Engineering
# --------------------------------------------
df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True
)
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = (
    df['Date']
    .dt
    .isocalendar()
    .week
    .astype(int)
)
df.drop(
    ['Date','Time'],
    axis=1,
    inplace=True
)
# Remove leakage
leakage_cols = [
    'Ride_Request_Count',
    'Surge_Multiplier',
    'Dynamic_Pricing_Activated',
    'Cancelled_Rides'
]
df.drop(
    columns=leakage_cols,
    inplace=True
)
# One-hot Encoding
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
df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)
# --------------------------------------------
# Model Training
# --------------------------------------------
X = df.drop(
    'Taxi_Demand',
    axis=1
)
Y = df['Taxi_Demand']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
m = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
m.fit(X_train,Y_train)
# --------------------------------------------
# Predictions
# --------------------------------------------
predictions = m.predict(X_test)
pred = predictions[0]
actual = Y_test.iloc[0]
# --------------------------------------------
# Demand Category
# --------------------------------------------
def demand_category(x):
    if x < 50:
        return "LOW"
    elif x < 100:
        return "MEDIUM"
    elif x < 200:
        return "HIGH"
    else:
        return "PEAK"
# --------------------------------------------
# Surge Pricing Recommendation
# --------------------------------------------
def surge_multiplier(x):
    if x < 50:
        return 1.0
    elif x < 100:
        return 1.2
    elif x < 200:
        return 1.5
    else:
        return 2.0
# --------------------------------------------
# Fleet Allocation
# --------------------------------------------
def allocate_drivers(demand):
    sedan = int(demand * 0.5)
    suv = int(demand * 0.3)
    luxury = int(demand * 0.2)
    return sedan,suv,luxury
sedan,suv,luxury = allocate_drivers(pred)
# --------------------------------------------
# RESULT CARD
# --------------------------------------------
print("\n")
print("="*60)
print("           SMART RIDE AI RESULT CARD")
print("="*60)
print(f"Predicted Taxi Demand     : {pred:.2f}")
print(f"Actual Taxi Demand        : {actual}")
print(f"Demand Category           : {demand_category(pred)}")
print(f"Recommended Surge Pricing : "f"{surge_multiplier(pred)}x")
print("-"*60)
print("Recommended Fleet Allocation")
print("-"*60)
print(f"Sedan Drivers : {sedan}")
print(f"SUV Drivers : {suv}")
print(f"Luxury Drivers : {luxury}")
print("="*60)
print("MODEL PERFORMANCE")
print("-"*60)
print("R2 Score :",round(r2_score(Y_test,predictions),4))
print("Train Score               :",round(m.score(X_train,Y_train),4))
print("Test Score                :",round(m.score(X_test,Y_test),4))
print("="*60)
