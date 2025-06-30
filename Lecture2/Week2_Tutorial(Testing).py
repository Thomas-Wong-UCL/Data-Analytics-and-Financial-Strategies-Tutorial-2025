pip install statsmodels numpy matplotlib

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# AR(1) Generator
def generate_ar1(n=101, phi=0.6, sigma=1.0, seed=42):
    np.random.seed(seed)
    eps = np.random.normal(loc=0, scale=sigma, size=n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = phi * X[t-1] + eps[t]
    return X

# AR(5) Generator
def generate_ar5(n=105, phi=[0.5, -0.3, 0.2, -0.1, 0.05], sigma=1.0, seed=42):
    np.random.seed(seed)
    eps = np.random.normal(loc=0, scale=sigma, size=n)
    X = np.zeros(n)
    for t in range(5, n):
        X[t] = (phi[0] * X[t-1] +
                phi[1] * X[t-2] +
                phi[2] * X[t-3] +
                phi[3] * X[t-4] +
                phi[4] * X[t-5] +
                eps[t])
    return X

# AR(7) Generator
def generate_ar7(n=107, phi=[0.4, -0.7, 0.3, -0.1, 0.5, 0.2, -0.4], sigma=1.0, seed=42):
    np.random.seed(seed)
    eps = np.random.normal(loc=0, scale=sigma, size=n)
    X = np.zeros(n)
    for t in range(7, n):
        X[t] = (phi[0] * X[t-1] +
                phi[1] * X[t-2] +
                phi[2] * X[t-3] +
                phi[3] * X[t-4] +
                phi[4] * X[t-5] +
                phi[5] * X[t-6] +
                phi[6] * X[t-7] +
                eps[t])
    return X

# Generate Datasets
dataset1 = generate_ar1(n=101, phi=0.6, sigma=1.0, seed=42)
phi_ar5 = [0.5, -0.3, 0.2, -0.1, 0.05]
dataset2 = generate_ar5(n=105, phi=phi_ar5, sigma=1.0, seed=42)
phi_ar7 = [-0.2, 0.0, 0.0, -0.0, 0.0, 0.0, 0.8]
dataset3 = generate_ar7(n=107, phi=phi_ar7, sigma=1.0, seed=42)

# prompt: can you save dataset1,dataset2 and dataset3 to css?

import pandas as pd

# Assuming dataset1, dataset2, and dataset3 are already defined as in your provided code.

# Create pandas DataFrames
df1 = pd.DataFrame({'dataset1': dataset1[1:]})
df2 = pd.DataFrame({'dataset2': dataset2[5:]})
df3 = pd.DataFrame({'dataset3': dataset3[7:]})

# Save to CSV files
df1.to_csv('dataset1.csv', index=False)
df2.to_csv('dataset2.csv', index=False)
df3.to_csv('dataset3.csv', index=False)

df = pd.read_csv('dataset3.csv')
print(df.head())
    # If your data has a different column name, replace 'value' with that name.
time_series = df
    # 2. Select the AR lag using the PACF
plot_pacf(time_series, lags=20)
plt.show()


model = ARIMA(time_series, order=(7, 0, 0))
fitted_model = model.fit()
    
# 4. Generate the h-steps-ahead forecast
h = 7  # adjust this as needed
forecast = fitted_model.forecast(steps=h)
    
print(f"\nFitted AR(p={7}) model summary:")
print(fitted_model.summary())
    
print(f"\n{h}-step ahead forecast:")
print(forecast)
