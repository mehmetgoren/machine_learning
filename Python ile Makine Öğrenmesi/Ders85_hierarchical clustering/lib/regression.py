"""
    regression algorithms by Gökalp Gören
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

    


def linear_regression(x, y, x_test):
    lr = LinearRegression()
    lr.fit(x, y) #fit === train
    predict = lr.predict(x_test)
    return predict



#bunu test et, çünkü x_test geldi. Sıkıntı çıkarsa önceki versiyonlarla bir karşılaştır.
def polynomial_regression(x, y, x_test, degree=5,include_bias=True):#this is a nonlinear model. The degree variable determinates the polynomial' s degree.
    poly_reg = PolynomialFeatures(degree=degree, include_bias=include_bias)
    x_poly = poly_reg.fit_transform(x)#ilk eleman B0(bias ve x^0 == 1 olduğundan hepsi 1)

    #yeniden Linear Regression. Buradaki 2 üstsel değerler ile (polinom) linear regression yapıyor.
    lr = LinearRegression()
    lr.fit(x_poly, y)
    predict = lr.predict(poly_reg.fit_transform(x_test))
    return predict
    


def svr_predict(x, y, x_test, scale=True, kernel="rbf"):
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    svr_regressor = SVR(kernel=kernel)
    svr_regressor.fit(x, y)#train
    predict = svr_regressor.predict(x_test)
    return predict



def decision_tree(x, y, x_test, random_state=0):
    tree_regressor = DecisionTreeRegressor(random_state=random_state)
    tree_regressor.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
    predict = tree_regressor.predict(x_test)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider
    return predict



def random_forest_regression(x, y, x_test, n_estimators=10, random_state=0):
    rf_reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)#n_estimators kaç tane decision tree çizileceği.
    rf_reg.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
    predict = rf_reg.predict(x_test)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider
    return predict
