"""
by Gökalp Gören
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

    

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