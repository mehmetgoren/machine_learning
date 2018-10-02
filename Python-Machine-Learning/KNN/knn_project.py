"""
    KNN Project
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("KNN_Project_Data")#index col ilk kolonu alma demek.
X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X) 



from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
kc = KNeighborsClassifier(n_neighbors=12, metric="minkowski")
kc.fit(X_train, y_train)
y_pred = kc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)



#modeli geliştirelim.
error_rates = []
for j in range(1,40):
    kc = KNeighborsClassifier(n_neighbors=j, metric="minkowski")
    kc.fit(X_train, y_train)
    y_pred = kc.predict(X_test)
    error_rates.append(np.mean(y_pred!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rates,color="blue", linestyle="--",marker="o",markerfacecolor="red",markersize=10)
plt.title("Error Rate vs K")
plt.xlabel("R")
plt.ylabel("Error Rate")