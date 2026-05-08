import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import os

os.makedirs("models", exist_ok=True)

print("Loading geological data...")
geo_df = pd.read_csv("data/geological_processed.csv")

feature_cols = ["Depth_m","Soil_pH","Silica_pct","Iron_Oxide_pct",
                "Sulfide_pct","Temperature_C","Pressure_atm","Rock_Type"]
X = geo_df[feature_cols]
y_grade = geo_df["Ore_Grade_g_t"]
y_viable = geo_df["Economically_Viable"]

X_train, X_test, y_train, y_test = train_test_split(X, y_grade, test_size=0.2, random_state=42)

# Ore Grade Regression
print("\nTraining Random Forest Regressor (Ore Grade)...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"  RMSE: {rmse:.4f}")
print(f"  R2 Score: {r2:.4f}")
joblib.dump(rf_reg, "models/rf_ore_grade.pkl")
print("  Model saved: rf_ore_grade.pkl")

# Viability Classification
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_viable, test_size=0.2, random_state=42)
print("\nTraining Random Forest Classifier (Viability)...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train2, y_train2)
y_pred2 = rf_clf.predict(X_test2)
acc = accuracy_score(y_test2, y_pred2)
print(f"  Accuracy: {acc*100:.2f}%")
joblib.dump(rf_clf, "models/rf_viability.pkl")
print("  Model saved: rf_viability.pkl")

print("\nRANDOM FOREST TRAINING COMPLETE!")