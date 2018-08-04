"""
    random forest classification
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("veriler.csv")

x = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)


ss = StandardScaler()
x_train = ss.fit_transform(x_train);
x_test = ss.transform(x_test)#fit demeye gerek yok zaten öğrendi.

rf = RandomForestClassifier(criterion="entropy", n_estimators=10)#gini de yapabilirsin. defauylt gini' dir.
rf.fit(x_train, y_train)
predict = rf.predict(x_test)
print(predict)
print(y_test.values)


#lets evaluate the success rate.
cm = confusion_matrix(y_test.values, predict)