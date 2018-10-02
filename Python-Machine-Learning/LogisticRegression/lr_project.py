"""
    Logistic Regression Project 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("advertising.csv")

#print(len(df["Country"].unique()))

sb.heatmap(df.corr(), cmap="coolwarm", annot=True)

X = df.iloc[:,[0,1,2,3,6]].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)