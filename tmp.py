import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# データの読み込み
data = pd.read_csv('patient.csv')
X = np.arange(len(data))[:,np.newaxis]
y = np.array(data['infected_persons'])

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
poly = PolynomialFeatures(degree=3).fit(X_scaled)
X1 = poly.transform(X_scaled)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X1, y)

print(f'predict: {lr.predict(poly.transform(scaler.transform(np.array([len(data) + 1])[:, np.newaxis])))}')

x = np.arange(0, 80)[:, np.newaxis]
x1 = poly.transform(scaler.transform(x))
plt.plot(X[:, 0], y)
plt.plot(x[:, 0], lr.predict(x1))
plt.show()