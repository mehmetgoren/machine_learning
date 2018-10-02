
import pandas as pd


df = pd.read_csv("iris.csv")


from sklearn.preprocessing import LabelEncoder
df["species"] = LabelEncoder().fit_transform(df.iloc[:,-1].values)
df["species"] = df["species"].apply(int)#verileri int e cast et.

#X = df.iloc[:,0:4].values
#y = df.iloc[:,-1].values

#veya
X = df.drop("species", axis=1)
y = df["species"]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)



import tensorflow as tf
feature_columns = []
for col in X.columns:
    feature_columns.append(tf.feature_column.numeric_column(col))


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,batch_size=10,num_epochs=5, shuffle=True)

classifier = tf.estimator.DNNClassifier(hidden_units=[30,20,10],n_classes=3, feature_columns=feature_columns)

classifier.train(input_fn=input_func, steps=50)

pred_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

predictions = classifier.predict(input_fn=pred_func)

y_pred = []
for pred in predictions:
    y_pred.append(pred["class_ids"][0])
    
    
from sklearn.metrics import confusion_matrix, classification_report as cr
classification_report = cr(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)