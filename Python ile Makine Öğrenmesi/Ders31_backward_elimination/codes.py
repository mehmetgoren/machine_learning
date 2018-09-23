"""
in this chapter, we calculate the p-value to evaluate which column has the most impact on the dataset.
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


#multiple linear regressin formula is y = B1*x1 + B2*x2 + B3*x3+ ... Bn*xn + B0 + Error
#lets' add the B0 value to the dataset
B0 = np.append(arr = np.ones((22,1)).astype(int), values=datatable_without_boy, axis=1)
datatable_without_boy_values = datatable_without_boy.iloc[:,[0,1,2,3,4,5]].values #bu array hangi kolonları aldığımızı gösteriyor.
boy = dataset.iloc[:,1:2].values

import statsmodels.formula.api as sm
result_ols = sm.OLS(endog=boy, exog=datatable_without_boy_values)
result = result_ols.fit()

#****************buradan çıkan sonuçta P>|t| değeri ne kadar küçükse / düşükse o kadar iyi. *********************************
print(result.summary())#çıkan x1,x2,x3,x4,x5,x6 değerleri [0,1,2,3,4,5] ' e karşılık geliyor.

#********** Backward elemination' a göre en yüksek p-value değerini elemeliyiz.
#bnu sonuçta 5 elemean P-value değeri en yük olduğu için onu çıkarıyoruz.




#burada 4. kolon (yaş) çıkıyor  ki backward-elemination yapıyoruz.
print("------------------------------------------------------------------------------------")
print("\n\n\n\n\n")

B0 = np.append(arr = np.ones((22,1)).astype(int), values=datatable_without_boy, axis=1)
datatable_without_boy_values = datatable_without_boy.iloc[:,[0,1,2,3,5]].values #bu array hangi kolonları aldığımızı gösteriyor.
boy = dataset.iloc[:,1:2].values

result_ols = sm.OLS(endog=boy, exog=datatable_without_boy_values)
result = result_ols.fit()

print(result.summary())



