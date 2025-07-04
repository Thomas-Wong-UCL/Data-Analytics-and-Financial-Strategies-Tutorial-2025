# -*- coding: utf-8 -*-
"""Lecture3(v0.2).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e6Y5caI95a-Htl_zhxubxMa2dmVdVWWc
"""

pip install statsmodels numpy matplotlib

import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

def generate_ma1(n=100, phi=0.5, theta=0.6, sigma=1.0, seed=43):
    np.random.seed(seed)
    a_t = np.random.normal(loc=0, scale=sigma, size=n)#random noise
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * (1 - theta) + a_t[t] - theta * a_t[t-1]
    return r

def generate_ma1_with_seasonality_trend(n=100, phi=0.5, theta=0.6, sigma=1.0, seed=43,
                                        a=1/7, trend_slope=0.1):
    """
    Generate an MA(1) time series with added seasonality and trend.

    Parameters:
    - n (int): Number of data points to generate.
    - phi (float): Coefficient for the constant term.
    - theta (float): MA(1) coefficient.
    - sigma (float): Standard deviation of the noise.
    - seed (int): Random seed for reproducibility.
    - a (float): Frequency parameter for seasonality (1/7 for weekly seasonality).
    - trend_slope (float): Slope of the linear trend.

    Returns:
    - r (np.ndarray): Generated time series with MA(1), seasonality, and trend.
    """
    np.random.seed(seed)

    # Generate white noise
    a_t = np.random.normal(loc=0, scale=sigma, size=n)

    # Initialize the time series array
    r = np.zeros(n)

    # Generate MA(1) process
    for t in range(1, n):
        r[t] = phi * (1 - theta) + a_t[t] - theta * a_t[t-1]

    # Time index
    t_array = np.arange(n)

    # Add seasonality: cosine function with frequency a
    seasonality = np.cos(2 * np.pi * a * t_array)

    # Add linear trend
    trend = trend_slope * t_array

    # Combine MA(1), seasonality, and trend
    r += seasonality + trend

    return r

dataset1_ma = generate_ma1(n=100, phi=0.6, theta=0.6, sigma=1.0, seed=43)
plt.plot(dataset1_ma)
dataset1_ma_with_seasonality_trend = generate_ma1_with_seasonality_trend(n=100, phi=0.6, theta=0.6, sigma=1.0, seed=43, a=1/7, trend_slope=0.1)
plt.plot(dataset1_ma_with_seasonality_trend)

print(dataset1_ma_with_seasonality_trend)
plot_acf(dataset1_ma_with_seasonality_trend, lags=20)
plot_pacf(dataset1_ma_with_seasonality_trend, lags=20)
plt.show()

frequencies, power_spectrum = signal.periodogram(dataset1_ma_with_seasonality_trend, fs=1)
plt.plot(frequencies, power_spectrum)
T=1/np.mean(frequencies)
T

result_ma_season_trend = seasonal_decompose(dataset1_ma_with_seasonality_trend, model='additive', period=20)
result_ma_season_trend.plot()
plt.show()

def generate_arma1_with_seasonality_trend(n=100, phi=[0.3, 0.4], theta=0.6, sigma=1.0, seed=43, a=1/7, trend_slope=0.1):
  np.random.seed(seed)

  # Generate white noise
  a_t = np.random.normal(loc=0, scale=sigma, size=n)

  # Initialize the time series array
  r = np.zeros(n)

  # Generate ARMA(1) process
  for t in range(1, n):
      r[t] = phi[0] + phi[1] * r[t-1] + phi[0] * (1 - theta) + a_t[t] - theta * a_t[t-1]

  # Time index
  t_array = np.arange(n)

  # Add seasonality: cosine function with frequency a
  seasonality = np.cos(2 * np.pi * a * t_array)

  # Add linear trend
  trend = trend_slope * t_array

  # Combine ARMA(1), seasonality, and trend
  r += seasonality + trend

  return r

dataset1_arma_with_seasonality_trend = generate_arma1_with_seasonality_trend(n=100, phi=[0.3, 0.4], theta=0.6, sigma=1.0, seed=43, a=1/7, trend_slope=0.1)
plt.plot(dataset1_arma_with_seasonality_trend)

print(dataset1_ma_with_seasonality_trend)
plot_acf(dataset1_arma_with_seasonality_trend, lags=20)
plot_pacf(dataset1_arma_with_seasonality_trend, lags=20)
plt.show()

frequencies, power_spectrum = signal.periodogram(dataset1_arma_with_seasonality_trend, fs=1)
plt.plot(frequencies, power_spectrum)
T=1/np.mean(frequencies)
T

result_arma_season_trend = seasonal_decompose(dataset1_arma_with_seasonality_trend, model='additive', period=20)
result_arma_season_trend.plot()
plt.show()