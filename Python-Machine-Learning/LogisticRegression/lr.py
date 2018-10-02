"""
    Logistic Regression
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df_train = pd.read_csv("titanic_train.csv")


##null verileri bulurken
#sb.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
#plt.show();
#
#sb.countplot(x="Survived", hue="Sex", data=df_train)
#plt.show()
#
#df_train["Age"].hist(bins=35)
#plt.show()
#
#
#df_train["Fare"].hist(bins=45, figsize=(10,4))
#plt.show()
#
#
#u=df_train["Embarked"].unique()
#print(u)

from sklearn.preprocessing import Imputer

X_train = df_train.iloc[:,:]
X_train.drop(["Cabin"],axis=1,inplace=True)#Cabin kolonunda çok fazla null veri var ve çıkarılması gerekli.
X_train.iloc[:,5] = np.squeeze(Imputer().fit_transform(X_train.iloc[:,5].values.reshape(-1,1)))#yaşa ortalama veriliyor.
X_train.dropna(inplace=True)#embark da null olanları silinsin (row bazlı).

sex = pd.get_dummies(df_train["Sex"], drop_first=True)
embark = pd.get_dummies(df_train["Embarked"], drop_first=True)

X_train = pd.concat([X_train,sex,embark], axis=1)

#şimdi gereksiz kolonları çıkaralım
X_train.drop(["PassengerId","Survived","Name","Sex","Ticket","Embarked"], axis=1, inplace=True)

y_train = df_train.iloc[:,1]





#şimdi de test' i hazırlayalım
df_test = pd.read_csv("titanic_test.csv")
X_test = df_test.iloc[:,:]
X_test.drop(["Cabin"],axis=1,inplace=True)#Cabin kolonunda çok fazla null veri var ve çıkarılması gerekli.
X_test.iloc[:,4] = np.squeeze(Imputer().fit_transform(X_test.iloc[:,4].values.reshape(-1,1)))#yaşa ortalama veriliyor.    
X_test.dropna(inplace=True)#embark da null olanları silinsin (row bazlı).
sex = pd.get_dummies(X_test["Sex"], drop_first=True)
embark = pd.get_dummies(X_test["Embarked"], drop_first=True)
X_test = pd.concat([X_test,sex,embark], axis=1)
X_test.drop(["PassengerId","Name","Sex","Ticket","Embarked"], axis=1, inplace=True)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



#sb.distplot(df["Yearly Amount Spent"])
#plt.show()
#sb.heatmap(corr, cmap="coolwarm", annot=True)
#plt.show()
#
#X = df.iloc[:,[3,4,6]].values
#y=df.iloc[:,[7]].values