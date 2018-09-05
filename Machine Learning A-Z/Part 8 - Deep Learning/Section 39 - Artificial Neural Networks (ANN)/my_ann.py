"""
    ANN
"""

import numpy
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split


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
X=X.values
y = dataset.iloc[:,-1].values

ss = StandardScaler()
X = ss.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state=0)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#adding hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))#column_num + 1 / 2

classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))#relu=rectifier activitaion faunction



#output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

#compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_train, batch_size=128, epochs=350)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > .5)


#lets compare

def f1_score(cm):
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[1][0]
    fn = cm[0][1]
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp/(tp+fp)
    score = 2*precision*recall/(precision+recall)
    return (accuracy, score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
ann = f1_score(cm)



#lets involve others

def test(classifier):
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred)
    return f1_score(cm)


from sklearn.svm import SVC
svc_linear = test(SVC(kernel="linear", random_state=0))
svc_rbf = test(SVC(kernel="rbf", random_state=0))


from sklearn.naive_bayes import GaussianNB
naive_bayes = test(GaussianNB())

from sklearn.linear_model import LogisticRegression
logistic_regression = test(LogisticRegression(random_state=0))

from sklearn.neighbors import KNeighborsClassifier
k_nearest_neighbors = test(KNeighborsClassifier(n_neighbors=5))


from sklearn.tree import DecisionTreeClassifier
decision_tree = test(DecisionTreeClassifier(random_state=0))


from sklearn.ensemble import RandomForestClassifier
random_forest = test(RandomForestClassifier(n_estimators=100, random_state=0))


