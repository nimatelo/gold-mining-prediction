import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

print("Loading datasets...")
gold_df = pd.read_csv("data/gold_prices.csv")
geo_df = pd.read_csv("data/geological_data.csv")
prod_df = pd.read_csv("data/production_data.csv")

# --- GOLD PRICE PREPROCESSING ---
print("\nPreprocessing gold price data...")
gold_df["Date"] = pd.to_datetime(gold_df["Date"])
gold_df = gold_df.sort_values("Date")
le = LabelEncoder()
gold_df["Market_Sentiment"] = le.fit_transform(gold_df["Market_Sentiment"])
scaler_gold = MinMaxScaler()
gold_scaled = scaler_gold.fit_transform(gold_df[["Gold_Price_USD"]])
joblib.dump(scaler_gold, "models/scaler_gold.pkl")
gold_df.to_csv("data/gold_prices_processed.csv", index=False)
print("Gold price data preprocessed!")

# --- GEOLOGICAL PREPROCESSING ---
print("\nPreprocessing geological data...")
le2 = LabelEncoder()
geo_df["Rock_Type"] = le2.fit_transform(geo_df["Rock_Type"])
geo_df = geo_df.drop("Sample_ID", axis=1)
scaler_geo = MinMaxScaler()
feature_cols = ["Depth_m","Soil_pH","Silica_pct","Iron_Oxide_pct",
                "Sulfide_pct","Temperature_C","Pressure_atm","Rock_Type"]
geo_df[feature_cols] = scaler_geo.fit_transform(geo_df[feature_cols])
joblib.dump(scaler_geo, "models/scaler_geo.pkl")
geo_df.to_csv("data/geological_processed.csv", index=False)
print("Geological data preprocessed!")

# --- PRODUCTION PREPROCESSING ---
print("\nPreprocessing production data...")
prod_df["Date"] = pd.to_datetime(prod_df["Date"])
prod_df = prod_df.drop("Date", axis=1)
scaler_prod = MinMaxScaler()
prod_cols = ["Equipment_Efficiency","Workers_Count","Rainfall_mm",
             "Energy_Cost_USD","Downtime_Days","Safety_Incidents",
             "Avg_Gold_Price_USD"]
prod_df[prod_cols] = scaler_prod.fit_transform(prod_df[prod_cols])
joblib.dump(scaler_prod, "models/scaler_prod.pkl")
prod_df.to_csv("data/production_processed.csv", index=False)
print("Production data preprocessed!")

print("\nALL PREPROCESSING COMPLETE!")