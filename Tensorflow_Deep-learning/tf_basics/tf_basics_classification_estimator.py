import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("pima-indians-diabetes.csv")
df = df.drop("Group", axis=1)

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


cols_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree', 'Age']

X[cols_to_normalize] = X[cols_to_normalize].apply(lambda x: (x-x.min()) / (x.max()-x.min() ) )#normalize ediyoruz.

tf_feature_columns = []
for c in cols_to_normalize:
    tf_feature_columns.append(tf.feature_column.numeric_column(c))
    
#X["Age"].hist(bin=35)
#plt.show()
    
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=tf_feature_columns,n_classes=2)

model.train(input_fn=train_input_fn, steps=1000)

test_input_dn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10,num_epochs=1, shuffle=False)

result = model.evaluate(test_input_dn)

y_pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)

y_pred = list(model.predict(y_pred_input_fn))


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=tf_feature_columns,n_classes=2)
dnn_model.train(input_fn=train_input_fn, steps=1000)
result2 = dnn_model.evaluate(test_input_dn)
y_pred2 = list(dnn_model.predict(y_pred_input_fn))