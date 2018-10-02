"""
    Decision Trees and Random Forest
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("kyphosis.csv")#index col ilk kolonu alma demek.


#sb.pairplot(df, hue="Kyphosis")

X = df.iloc[:,1:4].values
y = df.iloc[:,0].values


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)



from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)



#lets do some random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
classification_report_rfc= cr(y_test,y_pred_rfc)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)