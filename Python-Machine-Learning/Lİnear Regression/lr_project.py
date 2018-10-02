"""
    Linear Regression
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("Ecommerce Customers")
corr = df.corr()

sb.distplot(df["Yearly Amount Spent"])
plt.show()
sb.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()

X = df.iloc[:,[3,4,6]].values
y=df.iloc[:,[7]].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= .2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.coef_)#kitaplarda ayarlanan katsayılar b1,b2,b3.... bunlar.


from sklearn.metrics import explained_variance_score
result_variance = explained_variance_score(y_test, y_pred)#yüksek değer iyi 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)


#plt.scatter(X_test[:,2], y_test)
#plt.show()

plt.scatter(y_test, y_pred)
plt.show()
sb.distplot((y_test-y_pred))
plt.show()

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))