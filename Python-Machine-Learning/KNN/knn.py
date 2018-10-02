"""
    KNN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("Classified Data", index_col=0)#index col ilk kolonu alma demek.
X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X) 



from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
kc = KNeighborsClassifier(n_neighbors=10, metric="minkowski")
kc.fit(X_train, y_train)
y_pred = kc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)