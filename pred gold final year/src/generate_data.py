import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# DATASET 1: GOLD PRICE
n_days = 1500
dt = 1/252
mu = 0.06
sigma = 0.18
start_price = 1750.0
dates = pd.date_range(start="2019-01-01", periods=n_days, freq="B")
prices = [start_price]
for _ in range(n_days - 1):
    shock = np.random.normal(0, 1)
    price = prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
    prices.append(price)
prices = np.array(prices)
usd_index = 100 + np.cumsum(np.random.normal(0, 0.3, n_days))
oil_price = 70 + np.cumsum(np.random.normal(0, 0.8, n_days))
inflation = 2.5 + np.cumsum(np.random.normal(0, 0.05, n_days))
interest_rate = np.clip(1.5 + np.cumsum(np.random.normal(0, 0.02, n_days)), 0, 10)
market_sentiment = np.random.choice(["Bullish","Bearish","Neutral"], size=n_days, p=[0.4,0.3,0.3])
gold_price_df = pd.DataFrame({
    "Date": dates,
    "Gold_Price_USD": np.round(prices, 2),
    "USD_Index": np.round(usd_index, 2),
    "Oil_Price_USD": np.round(np.abs(oil_price), 2),
    "Inflation_Rate": np.round(inflation, 2),
    "Interest_Rate": np.round(interest_rate, 2),
    "Market_Sentiment": market_sentiment,
})
gold_price_df.to_csv("data/gold_prices.csv", index=False)
print("gold_prices.csv saved!")

# DATASET 2: GEOLOGICAL DATA
n_samples = 2000
depth = np.random.uniform(50, 800, n_samples)
soil_ph = np.random.uniform(4.5, 8.5, n_samples)
silica = np.random.uniform(20, 80, n_samples)
iron_oxide = np.random.uniform(5, 40, n_samples)
sulfide = np.random.uniform(0.1, 15, n_samples)
temperature = 25 + 0.03*depth + np.random.normal(0, 2, n_samples)
pressure = 1 + 0.01*depth + np.random.normal(0, 0.5, n_samples)
rock_type = np.random.choice(["Granite","Basalt","Schist","Quartzite","Diorite"],
                              n_samples, p=[0.3,0.2,0.2,0.2,0.1])
ore_grade = (0.008*depth - 0.15*soil_ph + 0.05*sulfide - 0.02*silica +
             np.random.normal(0, 0.5, n_samples))
ore_grade = np.clip(ore_grade, 0.1, 15.0)
viable = (ore_grade >= 2.0).astype(int)
geological_df = pd.DataFrame({
    "Sample_ID": [f"GS-{str(i+1).zfill(4)}" for i in range(n_samples)],
    "Depth_m": np.round(depth, 2),
    "Soil_pH": np.round(soil_ph, 2),
    "Silica_pct": np.round(silica, 2),
    "Iron_Oxide_pct": np.round(iron_oxide, 2),
    "Sulfide_pct": np.round(sulfide, 2),
    "Temperature_C": np.round(temperature, 2),
    "Pressure_atm": np.round(pressure, 2),
    "Rock_Type": rock_type,
    "Ore_Grade_g_t": np.round(ore_grade, 3),
    "Economically_Viable": viable,
})
geological_df.to_csv("data/geological_data.csv", index=False)
print("geological_data.csv saved!")

# DATASET 3: PRODUCTION DATA
n_months = 1200
prod_dates = pd.date_range(start="2014-01-01", periods=n_months, freq="MS")
equipment_efficiency = np.random.uniform(60, 98, n_months)
workers = np.random.randint(200, 600, n_months)
rainfall_mm = np.abs(np.random.normal(80, 30, n_months))
energy_cost_usd = np.random.uniform(40000, 120000, n_months)
downtime_days = np.random.randint(0, 10, n_months)
safety_incidents = np.random.poisson(1.5, n_months)

base_production = 500
trend = np.linspace(0, 100, n_months)
seasonality = 30 * np.sin(2 * np.pi * np.arange(n_months) / 12)
noise = np.random.normal(0, 10, n_months)

production_kg = np.abs(
    base_production +
    trend +
    seasonality +
    (equipment_efficiency - 60) * 3 +
    (workers / 100) * 10 -
    downtime_days * 15 +
    noise
)

monthly_price = gold_price_df.groupby(
    gold_price_df["Date"].dt.to_period("M"))["Gold_Price_USD"].mean().values
if len(monthly_price) < n_months:
    monthly_price = np.pad(monthly_price, (0, n_months - len(monthly_price)), mode="edge")
monthly_price = monthly_price[:n_months]

revenue_usd = production_kg * monthly_price * 32.15
operating_cost = energy_cost_usd * 12 + workers * 800
profit_usd = revenue_usd - operating_cost

production_df = pd.DataFrame({
    "Date": prod_dates,
    "Production_kg": np.round(production_kg, 2),
    "Equipment_Efficiency": np.round(equipment_efficiency, 2),
    "Workers_Count": workers,
    "Rainfall_mm": np.round(rainfall_mm, 2),
    "Energy_Cost_USD": np.round(energy_cost_usd, 2),
    "Downtime_Days": downtime_days,
    "Safety_Incidents": safety_incidents,
    "Avg_Gold_Price_USD": np.round(monthly_price, 2),
    "Revenue_USD": np.round(revenue_usd, 2),
    "Profit_USD": np.round(profit_usd, 2),
})
production_df.to_csv("data/production_data.csv", index=False)
print("production_data.csv saved!")
print("ALL DATASETS GENERATED SUCCESSFULLY!")