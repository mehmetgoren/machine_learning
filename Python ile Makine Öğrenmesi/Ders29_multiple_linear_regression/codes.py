"""
we predict sex variable from dataset
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

datatable = pd.concat([ulkeler, datatable], axis=1)

x_train, x_test, y_train, y_test = tts(datatable, cinsiyet, test_size=.33)


print(ulkeler)

#######################################################################################
regressor = LinearRegression(n_jobs = -1)
regressor.fit(x_train, y_train)#6 kolonlu 6 boyutlu uzayda bir lenar modeli çıkacak

predict = regressor.predict(x_test)


#aslında bunun için cart da kullanılabilir
result = []
index = 0
for y_item in y_test.values:
    predicted_item = predict[index]
    index += 1
    result.append(("erkek" if y_item >= 0.5 else "kadın","erkek" if predicted_item >= 0.5 else "kadın" ))
    
