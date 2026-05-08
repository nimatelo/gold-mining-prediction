import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_curve, auc)
import joblib
import os

os.makedirs("outputs", exist_ok=True)

print("Loading geological data...")
geo_df = pd.read_csv("data/geological_processed.csv")

feature_cols = ["Depth_m","Soil_pH","Silica_pct","Iron_Oxide_pct",
                "Sulfide_pct","Temperature_C","Pressure_atm","Rock_Type"]
X = geo_df[feature_cols]
y = geo_df["Economically_Viable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = joblib.load("models/rf_viability.pkl")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f0f1a")

ax1 = axes[0]
ax1.set_facecolor("#12121a")
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Not Viable","Viable"],
            yticklabels=["Not Viable","Viable"],
            ax=ax1, linewidths=2)
ax1.set_title("Confusion Matrix", color="#FFD700", fontsize=14, pad=15)
ax1.set_xlabel("Predicted", color="#aaaaaa")
ax1.set_ylabel("Actual", color="#aaaaaa")
ax1.tick_params(colors="#aaaaaa")

# ── Classification Report Bar Chart ──
ax2 = axes[1]
ax2.set_facecolor("#12121a")
report = classification_report(y_test, y_pred, output_dict=True)
categories = ["Not Viable", "Viable"]
precision = [report["0"]["precision"], report["1"]["precision"]]
recall    = [report["0"]["recall"],    report["1"]["recall"]]
f1        = [report["0"]["f1-score"],  report["1"]["f1-score"]]
x = np.arange(len(categories))
width = 0.25
ax2.bar(x - width, precision, width, label="Precision", color="#FFD700", alpha=0.85)
ax2.bar(x,         recall,    width, label="Recall",    color="#00FF88", alpha=0.85)
ax2.bar(x + width, f1,        width, label="F1-Score",  color="#4C9BFF", alpha=0.85)
ax2.set_title("Classification Metrics", color="#FFD700", fontsize=14, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, color="#aaaaaa")
ax2.tick_params(colors="#aaaaaa")
ax2.set_ylim(0, 1.1)
ax2.legend(labelcolor="#aaaaaa", facecolor="#12121a")
ax2.set_ylabel("Score", color="#aaaaaa")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333355")

# ── ROC Curve ──
ax3 = axes[2]
ax3.set_facecolor("#12121a")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color="#FFD700", linewidth=2,
         label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax3.plot([0,1],[0,1], color="#555555", linestyle="--", linewidth=1)
ax3.fill_between(fpr, tpr, alpha=0.1, color="#FFD700")
ax3.set_title("ROC Curve", color="#FFD700", fontsize=14, pad=15)
ax3.set_xlabel("False Positive Rate", color="#aaaaaa")
ax3.set_ylabel("True Positive Rate", color="#aaaaaa")
ax3.tick_params(colors="#aaaaaa")
ax3.legend(labelcolor="#aaaaaa", facecolor="#12121a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333355")

acc = accuracy_score(y_test, y_pred)
fig.suptitle(f"Random Forest — Viability Classification Results | Accuracy: {acc*100:.2f}%",
             color="#FFD700", fontsize=13, y=1.02)

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("confusion_matrix.png saved!")
print(f"\nAccuracy:  {acc*100:.2f}%")
print(f"AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Viable","Viable"]))
print("\nCONFUSION MATRIX COMPLETE!")