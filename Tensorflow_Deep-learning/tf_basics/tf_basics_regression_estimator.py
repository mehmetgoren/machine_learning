import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


X = np.linspace(.0,10.0,1000000)
noise = np.random.randn(len(X))

#y = mx+b;

y = (.5*X) + 5 + noise


feature_columns = tf.feature_column.numeric_column("x",shape=[1])
estimator = tf.estimator.LinearRegressor(feature_columns=[feature_columns])

from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)


input_fn = tf.estimator.inputs.numpy_input_fn({"x":X_train},y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x":X_train},y_train, batch_size=8, num_epochs=1000, shuffle=False)
test_input_fn = tf.estimator.inputs.numpy_input_fn({"x":X_test},y_test, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn, steps=1000)
test_metrics = estimator.evaluate(input_fn=test_input_fn, steps=1000)


#bu değerlerden loss' lar birbirine yakın olmalı. eğer arada fark varsa yüksek miktarda overfitting var demektir.
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({"x":brand_new_data}, shuffle=False)
y_pred = list(estimator.predict(input_fn=input_fn_predict))

