"""
	Polynomial Regression
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
plot(X,y, y)


#lets look linearregression first
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X,y)
y_pred=le.predict(X)
plot(X,y, y_pred)


#here it is polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)#artırdıkça dereceyi iyileşiyor predict
X_poly = poly_reg.fit_transform(X)

le2 = LinearRegression()
le2.fit(X_poly,y)
y_pred_poly = le2.predict(X_poly)
plot(X,y, y_pred_poly)



how_much = le.predict(6.5)
how_musch_more_accurate = le2.predict(poly_reg.fit_transform(6.5))