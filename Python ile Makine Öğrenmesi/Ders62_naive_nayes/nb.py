"""
naive bayes
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("veriler.csv")

#corr = dataset.corr()

x = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)

#ss = StandardScaler()
#x_train = ss.fit_transform(x_train);
#x_test = ss.transform(x_test)#fit demeye gerek yok zaten öğrendi.


gnb = GaussianNB()
gnb.fit(x_train, y_train)

predict = gnb.predict(x_test)
print(predict)
print(y_test.values)

#lets evaluate the success rate.
cm = confusion_matrix(y_test.values, predict)