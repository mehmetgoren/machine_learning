"""
    an example of tensorflow estimator
"""

from sklearn.datasets import load_wine

wine_data = load_wine()

columns = wine_data["feature_names"]
X = wine_data["data"]
y = wine_data["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=101)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#estimator
import tensorflow as tf
print(tf.VERSION)

feature_columns = [tf.feature_column.numeric_column("x",shape=[13])]


from tensorflow import estimator
deep_model = estimator.DNNClassifier(hidden_units=[13,13,13], feature_columns=feature_columns, n_classes=3
                                     , optimizer=tf.train.GradientDescentOptimizer(learning_rate=.01)) 


input_fn = estimator.inputs.numpy_input_fn(x={"x":X_train}, y=y_train,shuffle=True,batch_size=2,num_epochs=10)
deep_model.train(input_fn=input_fn, steps=500)

input_fn_evaluation=estimator.inputs.numpy_input_fn(x={"x":X_test}, shuffle=False)

y_pred = list(deep_model.predict(input_fn=input_fn_evaluation))
y_pred = [p["class_ids"][0] for p in y_pred]
import numpy as np
y_pred = np.asarray(y_pred)


from sklearn.metrics import confusion_matrix, classification_report
cr = classification_report(y_test, y_pred)