"""
    Linear Regression
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("USA_Housing.csv")

#df.info()
#desc = df.describe()
#corr = df.corr()
#sb.distplot(df["Price"])
#sb.heatmap(df.corr(), cmap="coolwarm", annot=True)

X = df.iloc[:,0:-2].values
y = df.iloc[:,-2].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= .2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.coef_)


from sklearn.metrics import explained_variance_score
result_variance = explained_variance_score(y_test, y_pred)#yüksek değer iyi 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

plt.scatter(y_test, y_pred)
plt.show()
sb.distplot((y_test-y_pred))