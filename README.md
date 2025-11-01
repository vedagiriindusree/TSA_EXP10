# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 1.11.2025
### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load avocado dataset
# -----------------------------
data = pd.read_csv('/content/avocado.csv')

# Convert Date column to datetime and sort
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')

# -----------------------------
# Plot the time series
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Total Volume'], color='blue')
plt.xlabel('Date')
plt.ylabel('Total Volume')
plt.title('Avocado Total Volume Time Series')
plt.grid()
plt.show()

# -----------------------------
# Check Stationarity
# -----------------------------
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Total Volume'])

# -----------------------------
# ACF and PACF plots
# -----------------------------
plot_acf(data['Total Volume'].dropna(), lags=40)
plt.title('Autocorrelation Function (ACF) - Total Volume')
plt.show()

plot_pacf(data['Total Volume'].dropna(), lags=40)
plt.title('Partial Autocorrelation Function (PACF) - Total Volume')
plt.show()

# -----------------------------
# Train-test split (80/20)
# -----------------------------
train_size = int(len(data) * 0.8)
train, test = data['Total Volume'][:train_size], data['Total Volume'][train_size:]

# -----------------------------
# SARIMA Model (p,d,q)x(P,D,Q,12)
# -----------------------------
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# -----------------------------
# Forecast and Evaluation
# -----------------------------
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# -----------------------------
# Plot Predictions vs Actual
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Total Volume')
plt.title('SARIMA Model Predictions - Avocado Total Volume')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="733" height="459" alt="image" src="https://github.com/user-attachments/assets/de324017-4251-4ad2-bef6-203f9a7864c0" />

<img width="274" height="98" alt="image" src="https://github.com/user-attachments/assets/f68dd9c4-34ee-4470-a5b1-3e919d7bcd21" />

<img width="494" height="358" alt="image" src="https://github.com/user-attachments/assets/7c6e7cca-35ce-4657-b513-1ca118afaf0f" />

<img width="479" height="359" alt="image" src="https://github.com/user-attachments/assets/cbd44d47-2c7c-4efe-971a-4bd96ef038d2" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
