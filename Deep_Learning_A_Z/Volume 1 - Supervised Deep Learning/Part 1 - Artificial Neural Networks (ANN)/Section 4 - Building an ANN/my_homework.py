"""
    Homos work
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


ohe = OneHotEncoder()
le = LabelEncoder()

dataset = pd.read_csv("Churn_Modelling.csv")



geo = dataset.iloc[:,4].values
geo = le.fit_transform(geo)
geo = ohe.fit_transform(geo.reshape(-1,1)).toarray()
geo = geo[:,[1,2]]#to avoid dummy variable trap.
geo = pd.DataFrame(data=geo, columns=["Germany","Spain"])

gender = dataset.iloc[:,5].values
gender = le.fit_transform(gender)
gender = pd.DataFrame(data=gender, columns=["Gender"])

X = dataset.iloc[:,[3,6,7,8,9,10,11,12]]
X = pd.concat([geo,gender,X], axis=1)
df = X.iloc[:,:]
corr = X.corr()
X=X.values
y = dataset.iloc[:,-1].values

ss = StandardScaler()
X = ss.fit_transform(X)



#ann starts here.
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#adding hidden layer
classifier.add(Dense(units=12, kernel_initializer="uniform", activation="relu", input_dim=11))#column_num + 1 / 2

classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))#relu=rectifier activitaion faunction

classifier.add(Dense(units=3, kernel_initializer="uniform", activation="relu"))

#output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

#compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X, y, batch_size=128, epochs=125)


#from sklearn.svm import SVC
#classifier = SVC(kernel="rbf", random_state=0)
#classifier.fit(X,y)

y_p = classifier.predict(X)
y_p = y_p >=.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_p)


X_test = np.array([0.0,0,1,600,40,3,60000,2,1,1,50000]).reshape(1,-1)
X_test = ss.transform(X_test)

y_pred = classifier.predict(X_test)
exited = y_pred >=.5