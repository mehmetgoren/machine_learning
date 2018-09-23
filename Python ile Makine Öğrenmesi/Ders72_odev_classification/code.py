import pandas as pd
#from sklearn.cross_validation import train_test_split
from lib import classification as c, evaluation as e

dataset = pd.read_excel("iris.xls")

x = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4]


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)



predict = c.logistic_regression(x, y, x)
cm_logistic_regression = e.confusion_matrix(y.values, predict)



predict = c.knn(x,y,x)
cm_knn = e.confusion_matrix(y.values, predict)



predict = c.svm(x,y,x, kernel="linear")
cm_svm_linear =  e.confusion_matrix(y.values, predict)



predict = c.svm(x,y,x, kernel="rbf")
cm_svm_rbf =  e.confusion_matrix(y.values, predict)



predict = c.naive_bayes(x,y,x)
cm_svm_naive_bayes =  e.confusion_matrix(y.values, predict)



predict = c.decision_tree(x,y,x)
cm_decision_tree =  e.confusion_matrix(y.values, predict)



predict = c.random_forest(x,y,x)
cm_random_forest=  e.confusion_matrix(y.values, predict)