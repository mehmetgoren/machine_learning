"""
we predict height variable from dataset
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LinearRegression


#Amaç kadın mı erkek mi tahmin etmek

le = LabelEncoder()
ohe = OneHotEncoder()

def categorical_to_numeric(data, use_ohe = True):
    data = le.fit_transform(data).reshape(-1,1)
    if (use_ohe):
        data = ohe.fit_transform(data).toarray()
    return data

def remove_column(dataset, left, right):
    left_dataframe = dataset.iloc[:,:left]
    right_dataframe = dataset.iloc[:,right:]
    return pd.concat([left_dataframe, right_dataframe], axis=1)

dataset = pd.read_csv("veriler.csv")

ulkeler = dataset.iloc[:,0:1].values
ulkeler = categorical_to_numeric(ulkeler)
ulkeler = pd.DataFrame(data=ulkeler, columns=["fr","tr","us"])



cinsiyet = dataset.iloc[:,-1]
cinsiyet = categorical_to_numeric(cinsiyet, False)
cinsiyet = cinsiyet[:,0]
cinsiyet = pd.DataFrame(data=cinsiyet,columns=["cinsiyet"])



datatable = dataset.iloc[:,1:4].values
datatable = pd.DataFrame(data=datatable, columns=["boy","kilo","yas"])



datatable = pd.concat([ulkeler, datatable, cinsiyet], axis=1)



datatable_without_boy = remove_column(datatable, 3, 4)
boy = dataset.iloc[:,1:2]



x_train, x_test, y_train, y_test = tts(datatable_without_boy, boy, test_size=.33)



regressor = LinearRegression(n_jobs=-1)
regressor.fit(x_train, y_train)
predict = regressor.predict(x_test)



result = []
index = 0
for y_item in y_test.values:
    predicted_item = predict[index]
    index += 1
    result.append((int(y_item), int(predicted_item)))
