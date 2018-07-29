"""
categorical values
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer as Imp #sci-kit learn bilimsel makina öğrenmesi imputer == töhmet, buradaki affetmek


dataset = pd.read_csv("missing_values.csv")

imputer = Imp(missing_values="NaN", strategy="mean", axis=0) # this is a strategy. Nan for null values, mean for avarages of numbers

yas = dataset.iloc[:,1:4].values#1 den 4 e kadar değerleri al ki bu verisetinde bunlar sayısal verilerdir.

#print(yas)
#print("-----------------------------------------")
#print("\n")

imputer = imputer.fit(yas[:,1:4])

clean_values= imputer.transform(yas[:,1:4])# :, 1 == take the first column

#print(clean_values)# in this dataset, NaN and mean_value has been replaced so we have an new dataset that does not cause a data noise.

ulke = dataset.iloc[:,0:1].values

print(ulke)

print("-----------------------------------------")
print("\n")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0]) #:,0 tahe the zero column. 

print(ulke)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()#ülkeyi önce LabelEncoderile çevirmen gerekli. First you need to fit_and transform via LabelEncoder to make OneHotEncoder usable.

print("-----------------------------------------")
print("\n")
print(ulke)