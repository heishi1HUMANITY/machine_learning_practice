import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# データの読み込み
data = pd.read_csv('patient.csv')
X = np.arange(len(data))[:,np.newaxis]
mu = X.mean()
sigma = X.std()
X1 = (X - mu) / sigma
y = np.array(data['infected_persons'])

to_matrix = lambda x: np.vstack([np.ones(x.shape[0]), x[:, 0], x[:, 0] ** 2, x[:, 0] ** 3]).T
Z = to_matrix(X1)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Z, y)

print('predict: ', lr.predict(to_matrix(np.array([(len(X) - mu) / sigma])[:, np.newaxis])))
x = np.arange(1, 75)[:, np.newaxis]
x = (x - mu) / sigma
plt.plot(X1, y)
plt.plot(x, lr.predict(to_matrix(x)))
plt.show()
