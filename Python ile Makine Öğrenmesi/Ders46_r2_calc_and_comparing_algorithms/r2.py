"""
    calculating r2 and comparing algorithms
"""

from sklearn.metrics import r2_score
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


dataset= pd.read_csv("maaslar.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values


def draw(x, y, predict):
    plt.scatter(x, y, color="red")
    plt.plot(x, predict, color="blue")
    plt.show()#refresh eder.
    

def standart_scaler(x, y):    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)
    return x_scaled, y_scaled
    


#*********** Algorithms ***************************
def linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x, y) #fit === train
    predict = lr.predict(x)
    return predict

def polynomial_regression(x, y, degree=5):#this is a nonlinear model. The degree variable determinates the polynomial' s degree.
    poly_reg = PolynomialFeatures(degree=degree, include_bias=True)
    x_poly = poly_reg.fit_transform(x)#ilk eleman B0(bias ve x^0 == 1 olduğundan hepsi 1)

    #yeniden Linear Regression. Buradaki 2 üstsel değerler ile (polinom) linear regression yapıyor.
    lr = LinearRegression()
    lr.fit(x_poly, y)
    predict = lr.predict(x_poly)
    return predict
    
def svr_predict(x, y, kernel="rbf"):    
    svr_regressor = SVR(kernel=kernel)
    svr_regressor.fit(x, y)#train
    predict = svr_regressor.predict(x)
    return predict


def decision_tree(x, y):
    tree_regressor = DecisionTreeRegressor(random_state=0)
    tree_regressor.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
    predict = tree_regressor.predict(x)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider
    return predict


def random_forest_regression(x, y):
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)#n_estimators kaç tane decision tree çizileceği.
    rf_reg.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
    predict = rf_reg.predict(x)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider
    return predict
#*********** Algorithms ***************************
    

print("\n\n\n")

print("Linear Regression R-Square        : ", r2_score(y, linear_regression(x, y)))#1 çıktı overfitting
print("\n\n\n")

print("Polynomial Regression R-Square    : ", r2_score(y, polynomial_regression(x, y)))#1 çıktı overfitting
print("\n\n\n")

#lets scale the values first. this is support vector and this algorithm is sensitive  outliar values.
x_scaled, y_scaled = standart_scaler(x, y)
print("Support Vector Regression R-Square: ", r2_score(y_scaled, svr_predict(x_scaled, y_scaled)))#1 çıktı overfitting
print("\n\n\n")


print("Decision Tree R-Square            : ", r2_score(y, decision_tree(x, y)))#1 çıktı overfitting
print("\n\n\n")


print("Random Forest R-Square            : ", r2_score(y, random_forest_regression(x, y)))#0.9704434230386582 güzel bir değer
print("\n\n\n")

#òlinear regression' ın kötü çıkmasının sebebi verile rpolinamsal olduğu için. linear veriler olsa en iyi o çıkabilirdi.
