"""
  classification algorithms by Gökalp Gören
"""

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def logistic_regression(x, y, x_test, scale=True):
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    lr = LogisticRegression()
    lr.fit(x, y)#train them
    return lr.predict(x_test)



def knn(x, y, x_test, scale=True, n_neighbors=1, metric="minkowski"):
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    algo = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    algo.fit(x, y)
    return algo.predict(x_test)



def svm(x, y, x_test, scale=True, kernel="linear"):
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    svc = SVC(kernel=kernel)
    svc.fit(x, y)
    return svc.predict(x_test)



def naive_bayes(x, y, x_test, scale=True):
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    gnb = GaussianNB()
    gnb.fit(x, y)
    return gnb.predict(x_test)



def decision_tree(x, y, x_test, scale=True, criterion="entropy"):#gini de yapabilirsin. defauylt gini' dir.
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    dc = DecisionTreeClassifier(criterion=criterion)
    dc.fit(x, y)
    return dc.predict(x_test)



def random_forest(x, y, x_test, scale=True, criterion="entropy", n_estimators=10):#gini de yapabilirsin. defauylt gini' dir.
    if scale:
        ss = StandardScaler()
        x = ss.fit_transform(x);
        x_test = ss.transform(x_test)
        
    rf = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators)
    rf.fit(x, y)
    return rf.predict(x_test)
    