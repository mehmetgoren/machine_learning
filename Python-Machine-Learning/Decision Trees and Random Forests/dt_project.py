"""
    Decision Trees and Random Forest Project
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("loan_data.csv")#index col ilk kolonu alma demek.
corr = df.corr()


#categorical features
final_data = pd.get_dummies(df,columns=["purpose"], drop_first=True)
#final_data = final_data.drop(["purpose"],axis=1)
#


#visulation
#plt.figure(figsize=(10,6))
#df[df["credit.policy"]==1]["fico"].hist(bins=25, color="blue", label="Credit Policy==1", alpha=.6)
#df[df["credit.policy"]==0]["fico"].hist(bins=25, color="red", label="Credit Policy==0", alpha=.6)
#plt.legend()
#plt.xlabel("Fico Score")
#
#plt.show()
#
#
#plt.figure(figsize=(11,6))
#sb.countplot(x="purpose", hue="not.fully.paid", data=df, palette="Set1")
#



#df = df.drop(["purpose"],axis=1)

c = np.arange(0,19, dtype=int)
c = np.delete(c, 12)

X = final_data.iloc[:,c].values
y = final_data.iloc[:,12].values

X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=256, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)


