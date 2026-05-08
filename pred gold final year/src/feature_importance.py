import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os

os.makedirs("outputs", exist_ok=True)

print("Loading models...")
rf_grade = joblib.load("models/rf_ore_grade.pkl")
rf_viable = joblib.load("models/rf_viability.pkl")
xgb = joblib.load("models/xgb_production.pkl")

# ── Feature Names ──
geo_features = ["Depth", "Soil pH", "Silica %",
                "Iron Oxide %", "Sulfide %",
                "Temperature", "Pressure", "Rock Type"]

prod_features = ["Equipment\nEfficiency", "Workers\nCount",
                 "Rainfall", "Energy\nCost", "Downtime\nDays",
                 "Safety\nIncidents", "Gold\nPrice"]

fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor="#0f0f1a")

def plot_importance(ax, importances, features, title, color):
    ax.set_facecolor("#12121a")
    indices = np.argsort(importances)
    colors = [color if i == indices[-1] else "#FFD700"
              for i in range(len(importances))]
    bars = ax.barh(range(len(importances)),
                   importances[indices],
                   color=[colors[i] for i in indices],
                   alpha=0.85, height=0.6)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([features[i] for i in indices],
                        color="#e8e8e8", fontsize=10)
    ax.set_title(title, color="#FFD700", fontsize=13,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    for i, (bar, imp) in enumerate(zip(bars,
                                        importances[indices])):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center',
                color='white', fontsize=9)

# Plot 1: RF Ore Grade
plot_importance(axes[0],
    rf_grade.feature_importances_,
    geo_features,
    "🪨 Ore Grade Prediction\n(Random Forest)",
    "#00FF88")

# Plot 2: RF Viability
plot_importance(axes[1],
    rf_viable.feature_importances_,
    geo_features,
    "✅ Mine Viability\n(Random Forest)",
    "#4C9BFF")

# Plot 3: XGBoost Production
plot_importance(axes[2],
    xgb.feature_importances_,
    prod_features,
    "⚙️ Production Forecast\n(XGBoost)",
    "#FF4C4C")

fig.suptitle("Feature Importance Analysis — Gold Mining Prediction System",
             color="#FFD700", fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("feature_importance.png saved!")

print("\n--- TOP FACTORS FOR ORE GRADE ---")
indices = np.argsort(rf_grade.feature_importances_)[::-1]
for i in indices:
    print(f"  {geo_features[i]}: {rf_grade.feature_importances_[i]:.4f}")

print("\n--- TOP FACTORS FOR PRODUCTION ---")
indices = np.argsort(xgb.feature_importances_)[::-1]
for i in indices:
    print(f"  {prod_features[i].replace(chr(10),' ')}: {xgb.feature_importances_[i]:.4f}")

print("\nFEATURE IMPORTANCE COMPLETE!")