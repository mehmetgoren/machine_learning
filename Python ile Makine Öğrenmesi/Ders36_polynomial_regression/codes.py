import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot(x, y, predict):
    print("------------------------------------------------------------")
    print("\n\n\n\n\n\n\n")
    plt.scatter(x, y, color="red")
    plt.plot(x, predict, color="blue")# it's very bad predict
    plt.show()
    
def linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x, y) #fit === train
    predict = lr.predict(x)
    return predict

def polynomial_regression(x, y, degree=2):#this is a nonlinear model. The degree variable determinates the polynomial' s degree.
    poly_reg = PolynomialFeatures(degree=degree, include_bias=True)
    x_poly = poly_reg.fit_transform(x)#ilk eleman B0(bias ve x^0 == 1 olduğundan hepsi 1)

    #yeniden Linear Regression. Buradaki 2 üstsel değerler ile (polinom) linear regression yapıyor.
    lr = LinearRegression()
    lr.fit(x_poly, y)
    predict = lr.predict(x_poly)
    return predict

def rmse(y, p):
    index = 0
    total = 0  
    for p_item in p:
        y_item = y[index]
        index += 1
        total += math.pow((p_item - y_item), 2)

    return math.sqrt(total / len(y)) 
    


dataset= pd.read_csv("maaslar.csv")

x = dataset.iloc[:,1].values.reshape(-1,1)
y = dataset.iloc[:,2].values.reshape(-1,1)


#lienar regressiom
predict = linear_regression(x, y)
plot(x, y, predict)
#
 


#polynomial regression
for i in range(10):
    degree = i + 2
    predict = polynomial_regression(x, y, degree)
    plot(x, y, predict)

    print("\n\n\n")
    accuracy = rmse(y, predict)
    print("degree: ", degree, "rmse: ", accuracy)
#
