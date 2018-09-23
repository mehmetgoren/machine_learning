"""
    XGBoost
"""

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
X=X.values
y = dataset.iloc[:,-1].values

ss = StandardScaler()
X = ss.fit_transform(X)


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state=0)



#lets do  some xgboost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > .5)


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
xgboost_score = f1_score(cm)



from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y = y_train, cv = 10)
mean = accuracies.mean()#çok iyi.
stddev = accuracies.std();#az iyi


