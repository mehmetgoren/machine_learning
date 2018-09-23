# -*- coding: utf-8 -*-
"""
Simple Linear Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts



dataset = pd.read_csv("Salary_Data.csv")
corr = dataset.corr()

X = dataset.iloc[:,0:1].values
y =dataset.iloc[:,1:].values



X_train, X_test, y_train, y_test = tts(X, y, test_size=.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


plt.title("Salary vs Expreriance (Training set)")
plt.xlabel("Years of Experiance")
plt.ylabel("Salary")
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.show()


print("\n\n\n")


plt.title("Salary vs Expreriance (Training set)")
plt.xlabel("Years of Experiance")
plt.ylabel("Salary")
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.show()


print("\n\n\n")


plt.title("Salary vs Expreriance (Training set)")
plt.xlabel("Years of Experiance")
plt.ylabel("Salary")
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue")
plt.show()