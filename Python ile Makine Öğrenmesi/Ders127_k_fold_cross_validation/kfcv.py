"""
    k-fold cross validation
"""

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

x_train, x_test, y_train, y_test = tts(x, y, test_size=.25, random_state = 0)#random_state= 0 yap her zaman.


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)




#şimdi de k-fold ile split edelim

from sklearn.model_selection import cross_val_score
#estimater: classfier burada
#cv = kaç katlamalı. Ayrıntılar Ders 127' de.
cvs=cross_val_score(estimator=classifier, X=x_train, y=y_train,cv=4)
mean = cvs.mean()
standart_sapma = cvs.std()