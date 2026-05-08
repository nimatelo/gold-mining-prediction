import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

print("Loading datasets...")
gold_df = pd.read_csv("data/gold_prices.csv")
geo_df = pd.read_csv("data/geological_data.csv")
prod_df = pd.read_csv("data/production_data.csv")

print("\n--- GOLD PRICE DATA ---")
print(gold_df.describe())

print("\n--- GEOLOGICAL DATA ---")
print(geo_df.describe())

print("\n--- PRODUCTION DATA ---")
print(prod_df.describe())

# Plot 1: Gold Price Over Time
plt.figure(figsize=(12,5))
plt.plot(gold_df["Gold_Price_USD"], color="gold")
plt.title("Gold Price Over Time")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("outputs/gold_price_trend.png")
plt.close()
print("gold_price_trend.png saved!")

# Plot 2: Ore Grade Distribution
plt.figure(figsize=(8,5))
sns.histplot(geo_df["Ore_Grade_g_t"], bins=40, color="orange")
plt.axvline(2.0, color="red", linestyle="--", label="Viability Line")
plt.title("Ore Grade Distribution")
plt.xlabel("Ore Grade (g/t)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/ore_grade_distribution.png")
plt.close()
print("ore_grade_distribution.png saved!")

# Plot 3: Monthly Production
plt.figure(figsize=(12,5))
plt.bar(range(len(prod_df)), prod_df["Production_kg"], color="gold")
plt.title("Monthly Gold Production (kg)")
plt.xlabel("Month")
plt.ylabel("Production (kg)")
plt.tight_layout()
plt.savefig("outputs/monthly_production.png")
plt.close()
print("monthly_production.png saved!")

# Plot 4: Correlation Heatmap
plt.figure(figsize=(10,8))
numeric_cols = geo_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Geological Data Correlation")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()
print("correlation_heatmap.png saved!")

print("\nEDA COMPLETE! Check outputs folder for charts.")