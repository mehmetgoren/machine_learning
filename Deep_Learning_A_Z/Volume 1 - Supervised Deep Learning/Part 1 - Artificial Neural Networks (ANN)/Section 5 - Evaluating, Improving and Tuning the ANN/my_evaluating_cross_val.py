"""
    cross_validation
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


#from sklearn.svm import SVC
#classifier = SVC(kernel="rbf", random_state=0)
#classifier.fit(X,y)


#ann starts here.
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(): 
    classifier = Sequential()
    #adding hidden layer
    classifier.add(Dense(units=12, kernel_initializer="uniform", activation="relu", input_dim=11))#column_num + 1 / 2
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))#relu=rectifier activitaion faunction 
    classifier.add(Dense(units=3, kernel_initializer="uniform", activation="relu"))
    
    #output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    #compiling the ANN
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return classifier


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)  
accuracies = cross_val_score(estimator=classifier, X=X,y=y, cv=16)
mean = accuracies.mean()
variance = accuracies.std()



#dropout 



#classifier.fit(X, y, batch_size=128, epochs=125)
#y_pred = classifier.predict(X)
#y_pred = y_pred >=.5
#
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y,y_pred)
