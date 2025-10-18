# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 18-10-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Load avocado dataset
# -----------------------------
data = pd.read_csv('/content/avocado.csv')

# Convert Date column to datetime and sort
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')

# Use only the Total Volume column
avocado_data = data[['Total Volume']]

print("Shape of the dataset:", avocado_data.shape)
print("First 10 rows of the dataset:")
print(avocado_data.head(10))

# -----------------------------
# Plot original Total Volume data
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(avocado_data['Total Volume'], label='Original Total Volume Data', color='blue')
plt.title('Original Avocado Total Volume Data')
plt.xlabel('Index')
plt.ylabel('Total Volume')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Moving Average (Rolling)
# -----------------------------
rolling_mean_5 = avocado_data['Total Volume'].rolling(window=5).mean()
rolling_mean_10 = avocado_data['Total Volume'].rolling(window=10).mean()

print("\nFirst 10 values (window=5):")
print(rolling_mean_5.head(10))

print("\nFirst 20 values (window=10):")
print(rolling_mean_10.head(20))

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(avocado_data['Total Volume'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='green')
plt.title('Moving Average of Avocado Total Volume')
plt.xlabel('Index')
plt.ylabel('Total Volume')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Exponential Smoothing
# -----------------------------
# Convert to monthly mean for smoother time series
data_monthly = data.set_index('Date')['Total Volume'].resample('MS').mean().dropna()

# Train-test split (80-20)
x = int(len(data_monthly) * 0.8)
train_data = data_monthly[:x]
test_data = data_monthly[x:]

# Fit model (Additive trend and Multiplicative seasonality)
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast for test data length
test_predictions = model.forecast(steps=len(test_data))

# Plot training, test, and forecast
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Train Data', color='blue')
plt.plot(test_data, label='Test Data', color='green')
plt.plot(test_predictions, label='Exponential Smoothing Predictions', color='red', linestyle='--')
plt.title('Exponential Smoothing - Avocado Total Volume')
plt.xlabel('Date')
plt.ylabel('Total Volume')
plt.legend()
plt.grid()
plt.show()

```
### OUTPUT:
Original Data:

<img width="344" height="302" alt="image" src="https://github.com/user-attachments/assets/53880943-c053-4cd0-91ab-871953694213" />

<img width="848" height="448" alt="image" src="https://github.com/user-attachments/assets/6650177d-b2a9-4eba-b36a-d5b8bae95562" />


Moving Average(Rolling):

<img width="260" height="200" alt="image" src="https://github.com/user-attachments/assets/6b2d10ed-7e40-4a7f-8ac5-12d2dd1e0a7d" />
<img width="255" height="340" alt="image" src="https://github.com/user-attachments/assets/000dda98-bda9-4a62-a752-9c47202d01a9" />


Plot Transform Dataset:

<img width="826" height="433" alt="image" src="https://github.com/user-attachments/assets/f9cb18cd-fbb4-4ca8-8435-8b82ee16007f" />


Exponential Smoothing:
<img width="843" height="452" alt="image" src="https://github.com/user-attachments/assets/d50d7f40-294c-45fc-afab-a36c3bedfd0d" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
