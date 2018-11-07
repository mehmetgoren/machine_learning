"""
    Tensorflow Linear Regression - Exercise
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cal_housing_clean.csv")

X = df.drop("medianHouseValue", axis=1)
y = df["medianHouseValue"]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X= pd.DataFrame(data=scaler.transform(X), columns=X.columns, index=X.index)


from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=101)



#tensorflow için kolonları oluştur
tf_columns = []
for c in X.columns:
    tf_columns.append(tf.feature_column.numeric_column(c))
#

#eğitim için gerekli parametreler
input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000,shuffle=True)
#

#modeli eğitelim
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=tf_columns)
model.train(input_fn=input_fn, steps=20000)
#

#tahmini gerçekleştirelim
y_pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
y_pred = list(model.predict(y_pred_input_func))
#


#şimdide değerlendirme (evaluate) yapalım
y_pred_as_number = []
for pred in y_pred:
    y_pred_as_number.append(pred["predictions"])

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_pred_as_number)**.5


#
#from sklearn.metrics import explained_variance_score
#result_variance = explained_variance_score(y_test, y_pred)#yüksek değer iyi 
#
#
#from sklearn.metrics import r2_score
#r2 = r2_score(y_test, y_pred)
#
#
##plt.scatter(X_test[:,2], y_test)
##plt.show()
#
##plt.scatter(y_test, y_pred)
##plt.show()
##sb.distplot((y_test-y_pred))
##plt.show()
#
#from sklearn import metrics
#print(metrics.mean_absolute_error(y_test,y_pred))
#print(metrics.mean_squared_error(y_test,y_pred))