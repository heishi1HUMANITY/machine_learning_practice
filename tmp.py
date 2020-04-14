import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

# データの読み込み
data = pd.read_csv('patient.csv')
X = np.arange(len(data))[:,np.newaxis]
y = np.array(data['total_confirmed_cases'])

# 曜日の追加
day = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'] 
days = pd.DataFrame([day[i % 7] for i in range(len(data) + 5)], columns=['day_of_the_week'])
days_dummied = pd.get_dummies(days)
X = np.hstack([X, np.array(days_dummied[:len(data)])])

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=5), Lasso(alpha=100))
pipe.fit(X, y)

print(f'prediction: {pipe.predict(np.hstack([np.array([len(data)]), np.array(days_dummied)[len(data)]]).reshape(1,8))}')

x = np.arange(0, len(data) + 5)[:, np.newaxis]
x = np.hstack([x, np.array(days_dummied)])

plt.plot(X[:, 0], y, label='patients')
plt.plot(x[:, 0], pipe.predict(x), linestyle='dashed', label='prediction')
plt.legend(loc='best')
plt.show()