import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Gold Mining Prediction System",
    page_icon="🏆",
    layout="wide"
)

st.markdown("""
<style>
h1, h2, h3 { color: #FFD700 !important; }
.stApp { background-color: #0a0a0f; color: #e8e8e8; }
.metric-card {
    background: #12121a;
    border: 1px solid #FFD70033;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: 900; color: #FFD700; }
.metric-label { font-size: 0.85rem; color: #aaaaaa; margin-top: 5px; }
.gold-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #FFD700, transparent);
    margin: 20px 0;
}
.prediction-box {
    background: #1a1a2e;
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin: 10px 0;
}
.stButton>button {
    background: linear-gradient(135deg, #B8860B, #FFD700);
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 25px;
    width: 100%;
}
section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #FFD70033;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-size: 2.5rem; color:#FFD700;'>⚜ GOLD MINING PREDICTION SYSTEM ⚜</h1>
    <p style='color: #aaaaaa;'>INTELLIGENT FORECASTING — FINAL YEAR PROJECT</p>
</div>
<div class='gold-divider'></div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_df = pd.read_csv(os.path.join(base, "data", "gold_prices.csv"))
    geo_df = pd.read_csv(os.path.join(base, "data", "geological_data.csv"))
    prod_df = pd.read_csv(os.path.join(base, "data", "production_data.csv"))
    return gold_df, geo_df, prod_df

@st.cache_resource
def load_models():
    models = {}
    try:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models["rf_grade"] = joblib.load(os.path.join(base, "models", "rf_ore_grade.pkl"))
        models["rf_viable"] = joblib.load(os.path.join(base, "models", "rf_viability.pkl"))
        models["xgb"] = joblib.load(os.path.join(base, "models", "xgb_production.pkl"))
        models["scaler_geo"] = joblib.load(os.path.join(base, "models", "scaler_geo.pkl"))
        models["scaler_prod"] = joblib.load(os.path.join(base, "models", "scaler_prod.pkl"))
    except Exception as e:
        st.warning(f"Some models not loaded: {e}")
    return models

gold_df, geo_df, prod_df = load_data()
models = load_models()

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard Overview",
    "📈 Gold Price Analysis",
    "🪨 Ore Grade Prediction",
    "⚙️ Production Forecast",
    "🤖 Model Performance"
])

st.sidebar.markdown("""
<div style='padding:10px; color:#888; font-size:0.8rem;'>
    <b style='color:#FFD700'>Models:</b><br>
    ✅ LSTM — Gold Price<br>
    ✅ Random Forest — Ore Grade<br>
    ✅ XGBoost — Production<br><br>
    <b style='color:#FFD700'>Datasets:</b><br>
    📄 1,500 price records<br>
    📄 2,000 geological samples<br>
    📄 120 production months
</div>
""", unsafe_allow_html=True)

def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    ax.set_facecolor("#12121a")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#FFD700")
    return fig, ax

if page == "📊 Dashboard Overview":
    st.markdown("## 📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${gold_df['Gold_Price_USD'].mean():,.0f}</div>
            <div class='metric-label'>Avg Gold Price (USD)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{geo_df['Ore_Grade_g_t'].mean():.2f}</div>
            <div class='metric-label'>Avg Ore Grade (g/t)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        viable_pct = geo_df['Economically_Viable'].mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{viable_pct:.1f}%</div>
            <div class='metric-label'>Viable Mine Sites</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{prod_df['Production_kg'].mean():.0f}</div>
            <div class='metric-label'>Avg Monthly Production (kg)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Gold Price Trend")
        fig, ax = dark_fig()
        ax.plot(gold_df["Gold_Price_USD"].values, color="#FFD700", linewidth=1.2)
        ax.fill_between(range(len(gold_df)), gold_df["Gold_Price_USD"].values, alpha=0.15, color="#FFD700")
        ax.set_title("Simulated Gold Price")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Ore Grade Distribution")
        fig, ax = dark_fig()
        ax.hist(geo_df["Ore_Grade_g_t"], bins=40, color="#FFD700", edgecolor="#0a0a0f", alpha=0.85)
        ax.axvline(2.0, color="#FF4C4C", linestyle="--", linewidth=2, label="Viability (2 g/t)")
        ax.set_title("Ore Grade Distribution")
        ax.set_xlabel("Ore Grade (g/t)")
        ax.legend(labelcolor="#aaaaaa", facecolor="#12121a")
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Monthly Production")
        fig, ax = dark_fig()
        ax.bar(range(len(prod_df)), prod_df["Production_kg"], color="#FFD700", alpha=0.8)
        ax.set_title("Monthly Gold Production (kg)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Production (kg)")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Economic Viability")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f0f1a")
        counts = geo_df["Economically_Viable"].value_counts()
        ax.pie(counts, labels=["Not Viable", "Viable"],
               colors=["#FF4C4C", "#00FF88"],
               autopct="%1.1f%%",
               textprops={"color": "#e8e8e8"},
               startangle=90)
        ax.set_title("Mine Site Viability", color="#FFD700")
        fig.patch.set_facecolor("#0f0f1a")
        st.pyplot(fig)
        plt.close()

elif page == "📈 Gold Price Analysis":
    st.markdown("## 📈 Gold Price Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${gold_df['Gold_Price_USD'].min():,.0f}</div>
            <div class='metric-label'>Minimum Price</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${gold_df['Gold_Price_USD'].max():,.0f}</div>
            <div class='metric-label'>Maximum Price</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${gold_df['Gold_Price_USD'].std():,.0f}</div>
            <div class='metric-label'>Std Deviation</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    fig, ax = dark_fig((14, 5))
    ax.plot(gold_df["Gold_Price_USD"].values, color="#FFD700", linewidth=1.5, label="Gold Price")
    ma30 = gold_df["Gold_Price_USD"].rolling(30).mean()
    ma90 = gold_df["Gold_Price_USD"].rolling(90).mean()
    ax.plot(ma30.values, color="#00FF88", linewidth=1.2, linestyle="--", label="30-Day MA")
    ax.plot(ma90.values, color="#FF4C4C", linewidth=1.2, linestyle="--", label="90-Day MA")
    ax.fill_between(range(len(gold_df)), gold_df["Gold_Price_USD"].values, alpha=0.1, color="#FFD700")
    ax.set_title("Gold Price with Moving Averages")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.legend(labelcolor="#aaaaaa", facecolor="#12121a")
    st.pyplot(fig)
    plt.close()

    st.markdown("### LSTM Model Results")
    col1, col2, col3, col4 = st.columns(4)
    for col, (label, val) in zip([col1,col2,col3,col4],
        [("R² Score","0.8596"),("RMSE","143.66"),("Epochs","10"),("Sequence","60 days")]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

elif page == "🪨 Ore Grade Prediction":
    st.markdown("## 🪨 Ore Grade Prediction")
    st.markdown("Enter geological parameters below to predict ore grade.")
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        depth = st.slider("Depth (m)", 50, 800, 300)
        soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.5)
        silica = st.slider("Silica %", 20.0, 80.0, 45.0)
    with col2:
        iron_oxide = st.slider("Iron Oxide %", 5.0, 40.0, 15.0)
        sulfide = st.slider("Sulfide %", 0.1, 15.0, 5.0)
        temperature = st.slider("Temperature (°C)", 30.0, 50.0, 35.0)
    with col3:
        pressure = st.slider("Pressure (atm)", 1.5, 9.0, 4.0)
        rock_type = st.selectbox("Rock Type", ["Granite","Basalt","Schist","Quartzite","Diorite"])
        rock_map = {"Granite":1,"Basalt":0,"Schist":3,"Quartzite":2,"Diorite":4}

    if st.button("🔮 PREDICT ORE GRADE"):
        if "rf_grade" in models:
            features = np.array([[depth, soil_ph, silica, iron_oxide,
                                   sulfide, temperature, pressure, rock_map[rock_type]]])
            features_scaled = models["scaler_geo"].transform(features)
            grade = models["rf_grade"].predict(features_scaled)[0]
            viable = models["rf_viable"].predict(features_scaled)[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class='prediction-box'>
                    <div style='color:#aaa;'>PREDICTED ORE GRADE</div>
                    <div style='font-size:3rem; color:#FFD700; font-weight:900;'>{grade:.3f}</div>
                    <div style='color:#aaa;'>grams per tonne (g/t)</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                color = "#00FF88" if viable else "#FF4C4C"
                status = "✅ ECONOMICALLY VIABLE" if viable else "❌ NOT VIABLE"
                st.markdown(f"""<div class='prediction-box'>
                    <div style='color:#aaa;'>VIABILITY STATUS</div>
                    <div style='font-size:1.5rem; color:{color}; font-weight:900; margin-top:15px;'>{status}</div>
                    <div style='color:#aaa; margin-top:10px;'>Threshold: 2.0 g/t</div>
                </div>""", unsafe_allow_html=True)

elif page == "⚙️ Production Forecast":
    st.markdown("## ⚙️ Mine Production Forecast")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{prod_df['Production_kg'].mean():.0f} kg</div>
            <div class='metric-label'>Avg Monthly Production</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{prod_df['Production_kg'].sum():,.0f} kg</div>
            <div class='metric-label'>Total Production</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${prod_df['Profit_USD'].mean()/1e6:.1f}M</div>
            <div class='metric-label'>Avg Monthly Profit</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        efficiency = st.slider("Equipment Efficiency (%)", 60.0, 98.0, 80.0)
        workers = st.slider("Workers Count", 200, 600, 400)
        rainfall = st.slider("Rainfall (mm)", 10.0, 200.0, 80.0)
    with col2:
        energy_cost = st.slider("Energy Cost (USD)", 40000, 120000, 80000)
        downtime = st.slider("Downtime Days", 0, 10, 2)
        gold_price = st.slider("Gold Price (USD)", 1500, 5000, 2500)

    if st.button("🔮 PREDICT PRODUCTION"):
        if "xgb" in models:
            features = np.array([[efficiency, workers, rainfall,
                                   energy_cost, downtime, 1, gold_price]])
            features_scaled = models["scaler_prod"].transform(features)
            production = models["xgb"].predict(features_scaled)[0]
            st.markdown(f"""<div class='prediction-box' style='max-width:500px; margin:20px auto;'>
                <div style='color:#aaa;'>PREDICTED MONTHLY PRODUCTION</div>
                <div style='font-size:3.5rem; color:#FFD700; font-weight:900;'>{abs(production):.1f}</div>
                <div style='color:#aaa;'>kilograms of gold</div>
                <div style='color:#00FF88; margin-top:10px;'>
                    Est. Revenue: ${abs(production) * gold_price * 32.15:,.0f} USD
                </div>
            </div>""", unsafe_allow_html=True)

elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>
            <div style='color:#FFD700; font-size:1.1rem; margin-bottom:10px;'>🧠 LSTM</div>
            <div style='color:#aaa; font-size:0.85rem;'>Gold Price Forecasting</div>
            <div class='metric-value' style='font-size:1.5rem;'>R² = 0.8596</div>
            <div style='color:#00FF88;'>RMSE = 143.66</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div style='color:#FFD700; font-size:1.1rem; margin-bottom:10px;'>🌲 RANDOM FOREST</div>
            <div style='color:#aaa; font-size:0.85rem;'>Ore Grade Prediction</div>
            <div class='metric-value' style='font-size:1.5rem;'>R² = 0.9118</div>
            <div style='color:#00FF88;'>Accuracy = 94.25%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div style='color:#FFD700; font-size:1.1rem; margin-bottom:10px;'>⚡ XGBOOST</div>
            <div style='color:#aaa; font-size:0.85rem;'>Production Forecasting</div>
            <div class='metric-value' style='font-size:1.5rem;'>RMSE = 39.06</div>
            <div style='color:#00FF88;'>100 Estimators</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    fig, ax = dark_fig((10, 5))
    names = ["LSTM\n(Gold Price)", "Random Forest\n(Ore Grade)", "XGBoost\n(Production)"]
    scores = [0.8596, 0.9118, 0.75]
    colors = ["#FFD700", "#00FF88", "#4C9BFF"]
    bars = ax.bar(names, scores, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0.8, color="#FF4C4C", linestyle="--", linewidth=1.5, label="Good threshold (0.8)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{score:.4f}', ha='center', va='bottom', color='white', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("R² Score Comparison")
    ax.set_ylabel("R² Score")
    ax.legend(labelcolor="#aaaaaa", facecolor="#12121a")
    st.pyplot(fig)
    plt.close()

st.markdown("""
<div class='gold-divider'></div>
<div style='text-align:center; padding:15px; color:#555; font-size:0.8rem;'>
    ⚜ Gold Mining Prediction System — Final Year Project ⚜
</div>
""", unsafe_allow_html=True)
