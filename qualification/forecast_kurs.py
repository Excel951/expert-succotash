from csv import excel

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

SEED=42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

time_step = []
temps = []

data=pd.read_csv('train.csv')
data.head()

df = data.drop(['Bahan_pangan', 'Daerah', 'Harga_lag_1', 'Harga', 'Harga_lag_7', 'Harga_lag_30', 'Google_trends'], axis=1)
df.head()

df = df.drop_duplicates(ignore_index=True)

df.drop(index=120, inplace=True)

for index, value in enumerate(df['USDIDR'].values):
    try:
        temps.append(round(float(value), 2))
        time_step.append(index)
    except Exception as e:
        print(f'At index {index}')
        print(e)

# Drop rows with missing values (if any)
df = df[df['USDIDR'] != 0.0]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['USDIDR']].values)

# Create a windowed dataset
def create_windowed_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 2  # Use the last 2 days to predict the next day
X, y = create_windowed_dataset(scaled_data, window_size)

# Split into train and validation sets
split_index = int(len(X) * 0.8)
X_train, X_valid = X[:split_index], X[split_index:]
y_train, y_valid = y[:split_index], y[split_index:]

# Step 2: Build the Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(window_size,)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mape'])

# Step 3: Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=200,
    batch_size=2,
    verbose=1
)

# Step 4: Forecast Future Values
# Use the last `window_size` values from the dataset to predict the next value
last_window = scaled_data[-window_size:].reshape(1, -1)
predicted_scaled = model.predict(last_window)

# Inverse transform the prediction to get the actual value
predicted_value = scaler.inverse_transform(predicted_scaled)
print(f"Predicted USD/IDR: {predicted_value[0][0]:.2f}")