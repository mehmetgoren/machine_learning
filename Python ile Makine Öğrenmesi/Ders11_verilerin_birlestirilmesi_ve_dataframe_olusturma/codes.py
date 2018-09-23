"""
Ders11_verilerin_birlestirilmesi_ve_dataframe_olusturma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder #sci-kit learn bilimsel makina öğrenmesi imputer == töhmet, buradaki affetmek


dataset = pd.read_csv("missing_values.csv")


#böldük
numeric_values = dataset.iloc[:,1:4].values#1 den 4 e kadar değerleri al ki bu verisetinde bunlar sayısal verilerdir.
print(numeric_values)

print("\n\n\n")
print("-----------------------------------------------")

#nan değerleri ortalamaya çevirdik.
#[:,1:4]' e gerek yok.
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) # this is a strategy. Nan for null values, mean for avarages of numbers
imputer = imputer.fit(numeric_values[:,1:4])
numeric_values[:,1:4]= imputer.transform(numeric_values[:,1:4])# :, 1 == take the first column

print(numeric_values)

print("\n\n\n")
print("-----------------------------------------------")


#ülkeleri 1 ve 0 matrix' e çebireceğiz.
#burada nominal olan bir değeri sayısal veriye çeviriyoruz.
ulke = dataset.iloc[:,0:1].values

#harf den sayıya çevir.
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0]) #:,0 the the zero column. 

#çevirilen sayıları 0,1 matrix' e çevir.
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()#ülkeyi önce LabelEncoderile çevirmen gerekli. First you need to fit_and transform via LabelEncoder to make OneHotEncoder usable.

ulke_as_dataframe = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])

mumeric_values_as_dataframe = pd.DataFrame(data=numeric_values, index=range(22), columns=["boy","kilo","yas"])


print("\n\n\n")
print("-----------------------------------------------")


cinsiyet = dataset.iloc[:,-1].values # === cinsiyet = dataset.iloc[:,4:5].values
cinsiyet = pd.DataFrame(data=cinsiyet, index=range(22), columns=["sex"])
#cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])
print(cinsiyet)
#sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])#range=22 demek 1' den 22' ye kadar olan sayılar
#print(sonuc)
#print("\n\n\n")
#print("-----------------------------------------------")
#
#
#sonuc2 = pd.DataFrame(data=numeric_values,  index=range(22), columns=["boy","kilo","yaş"])
#print(sonuc2)
#print("\n\n\n")
#print("-----------------------------------------------")
#
#
#cinsiyet = dataset.iloc[:,4].values
##cinsiyet = le.fit_transform(cinsiyet)
#print(cinsiyet)

#concat = pd.concat([ulke_as_dataframe, mumeric_values_as_dataframe])#kartezyen çarpım gibi birleştiriyor.
concat = pd.concat([ulke_as_dataframe, mumeric_values_as_dataframe], axis=1)
concat = pd.concat([concat, cinsiyet], axis=1)