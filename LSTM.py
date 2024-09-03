import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the training data
train_data = pd.read_csv('train.csv', delimiter=';')
train_data = train_data.replace({',': '.'}, regex=True)
train_data = train_data.astype(float)

# Load the test data
test_data = pd.read_csv('test.csv', delimiter=';')
test_data = test_data.replace({',': '.'}, regex=True)
test_data = test_data.astype(float)

# Load the true SOC data
true_soc_data = pd.read_csv('test-compare-trueSOC.csv', delimiter=';')
true_soc_data = true_soc_data.replace({',': '.'}, regex=True)
true_soc_data = true_soc_data.astype(float)

# Preprocess the data
# Extract the features and the target
X_train = train_data[['Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time']].values
y_train = train_data['SOC'].values

X_test = test_data[['Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time']].values
true_soc = true_soc_data['SOC'].values
time = test_data['Time'].values

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM [samples, time steps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, batch_size=64, epochs=20, validation_split=0.2)

# Predict on test data
predicted_soc = model.predict(X_test_scaled)

# Plot the predicted SOC and true SOC over time
plt.figure(figsize=(12, 6))
plt.plot(time, predicted_soc, color='blue', label='Predicted SOC')
plt.plot(time, true_soc, color='red', linestyle='--', label='True SOC')
plt.title('SOC Prediction vs. True SOC Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('SOC')
plt.legend()
plt.show()
