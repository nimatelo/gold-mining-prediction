import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

print("Loading production data...")
prod_df = pd.read_csv("data/production_data.csv")

feature_cols = ["Equipment_Efficiency", "Workers_Count", "Rainfall_mm",
                "Energy_Cost_USD", "Downtime_Days", "Safety_Incidents",
                "Avg_Gold_Price_USD"]
X = prod_df[feature_cols]
y = prod_df["Production_kg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining XGBoost Regressor (Production)...")
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100)

y_pred = xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n  RMSE: {rmse:.4f}")
print(f"  R2 Score: {r2:.4f}")
joblib.dump(xgb, "models/xgb_production.pkl")
print("  Model saved: xgb_production.pkl")
print("\nXGBOOST TRAINING COMPLETE!")