"""
    SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)



from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)


#parametre seçimiş inanılmaz etkiledi.
from sklearn.grid_search import GridSearchCV
params = {"C":[.1,1,10,100,1000], "gamma":[1,.1,.01,.001,.0001]}
gc = GridSearchCV(SVC(),params, verbose=3, n_jobs=1)
gc.fit(X_train, y_train)
best_params = gc.best_params_
y_pred = gc.predict(X_test)
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
