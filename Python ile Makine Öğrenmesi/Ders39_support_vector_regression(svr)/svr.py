import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def scale(x, y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)
    return x_scaled, y_scaled

def svr_predict(x, y, kernel="rbf"):
    svr_regressor = SVR(kernel=kernel)
    svr_regressor.fit(x, y)#train
    predict = svr_regressor.predict(x)
    return predict

def plot(x, y, predict):
    plt.scatter(x, y, color="red")
    plt.plot(x, predict, color="blue")
    

dataset= pd.read_csv("maaslar.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values


x_scaled, y_scaled = scale(x, y)# svr da mutlaka scalling yapılmalı çünkü uç değerlere zaafiyeti var.


#predict = svr_predict(x_scaled, y_scaled, "linear")
#plot(x_scaled, y_scaled, predict)
#
#predict = svr_predict(x_scaled, y_scaled, "poly")
#plot(x_scaled, y_scaled, predict)

predict = svr_predict(x_scaled, y_scaled, "rbf")
plot(x_scaled, y_scaled, predict)

#predict = svr_predict(x_scaled, y_scaled, "sigmoid")
#plot(x_scaled, y_scaled, predict)
#
#predict = svr_predict(x_scaled, y_scaled, "precomputed")
#plot(x_scaled, y_scaled, predict)

