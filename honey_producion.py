
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = df['year']
X = X.values.reshape(-1, 1)

y = df['totalprod']

plt.scatter(X, y)

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_[0])

y_predict = regr.predict(X)
plt.plot(y_predict, X)
plt.show()

future_predict = regr.predict(y_predict)



