"""
    SVR
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Position_Salaries.csv")

corr = dataset.corr()

def plot(independent, dependent, predict):
    plt.scatter(independent,dependent, color="red")
    plt.plot(independent, predict, color="blue")
    plt.xlabel("Level")
    plt.ylabel("Salary")
    plt.show()


X = dataset.iloc[:,1:2]
y = dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)


scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X,y)


y_pred=scaler_y.inverse_transform(regressor.predict(scaler_X.transform(np.array([[6.5]]))))


plot(X,y,regressor.predict(X))