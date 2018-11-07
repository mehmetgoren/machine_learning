"""
    Tensorflow Saving and Restoing a Model
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("census_data.csv")

print(df["income_bracket"].unique())

def label_fix(label):
    return 0 if label == " <=50K" else 1

df["income_bracket"] = df["income_bracket"].apply(label_fix)


X = df.drop("income_bracket", axis=1)
y = df["income_bracket"]


from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size=.2, random_state=101)



#şimdi kategorik kolonları işleyelim
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",["Female","Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)


#şimdi de numeric kolonları oluşturalım
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")



feature_columns = [gender, occupation,marital_status,relationship,education,workclass,native_country,age,education_num,capital_gain,capital_loss,hours_per_week]


input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feature_columns)
model.train(input_fn=input_fn, steps=10000)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
y_pred = list(model.predict(input_fn=pred_fn))

y_pred = [pred["class_ids"][0] for pred in y_pred]



#şimdi de değerlendirelim

from sklearn.metrics import classification_report, confusion_matrix
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)


#şimdi modeli kayıt edeceğiz.
saver = tf.train.Saver()