import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# データの読み込み
data = pd.read_csv('patient.csv')
X = np.arange(len(data))[:,np.newaxis]
y = np.array(data['total_confirmed_cases'])

# 日付の追加
day = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'] 
days = pd.DataFrame([day[i % 7] for i in range(len(data) + 5)], columns=['day_of_the_week'])
days_dummied = pd.get_dummies(days)

X = np.hstack([X, np.array(days_dummied[:len(data)])])
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
poly = PolynomialFeatures(degree=5).fit(X_scaled)
X1 = poly.transform(X_scaled)

from sklearn.linear_model import Lasso
lr = Lasso(alpha=100).fit(X1, y)

print(f'prediction: {lr.predict(poly.transform(scaler.transform(np.hstack([np.array([len(data)]), np.array(days_dummied)[len(data)]]).reshape(1,8))))}')

x = np.arange(0, len(data) + 5)[:, np.newaxis]
x = np.hstack([x, np.array(days_dummied)])
x1 = poly.transform(scaler.transform(x))

plt.plot(X[:, 0], y, label='patients')
plt.plot(x[:, 0], lr.predict(x1), linestyle='dashed', label='prediction')
plt.legend(loc='best')
plt.show()