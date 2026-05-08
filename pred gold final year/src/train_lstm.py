import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

print("Loading gold price data...")
gold_df = pd.read_csv("data/gold_prices.csv")
gold_df["Date"] = pd.to_datetime(gold_df["Date"])
gold_df = gold_df.sort_values("Date")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(gold_df[["Gold_Price_USD"]])
joblib.dump(scaler, "models/scaler_lstm.pkl")

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled, SEQ_LENGTH)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("\nBuilding improved LSTM model...")
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True,
                       input_shape=(SEQ_LENGTH, 1))),
    Dropout(0.2),
    Bidirectional(LSTM(100, return_sequences=True)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

print("\nTraining LSTM with 50 epochs...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)
print(f"\n  RMSE:     {rmse:.4f}")
print(f"  R2 Score: {r2:.4f}")

model.save("models/lstm_gold_price.h5")
print("  Model saved: lstm_gold_price.h5")
print("\nLSTM TRAINING COMPLETE!")