"""
    Dropout
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

#droupout regularization to reduce overfitting - Disabling neuron randomly.
from keras.layers import Dropout

classifier = Sequential()
#adding hidden layer
classifier.add(Dense(units=12, kernel_initializer="uniform", activation="relu", input_dim=11))#column_num + 1 / 2
classifier.add(Dropout(p=.1))#p is the number that howmany neoruens will be disabled.

classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))#relu=rectifier activitaion faunction
classifier.add(Dropout(p=.2))

classifier.add(Dense(units=3, kernel_initializer="uniform", activation="relu"))
#classifier.add(Dropout(p=.1))

#output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

#compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X, y, batch_size=8, epochs=4)


y_p = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y>=.5,y_p >=.5)