"""
in this chapter, we calculate the p-value to evaluate which column has the most impact on the dataset.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

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

#backward elamination için.
def ols_summary(ones_length, x_dataset, y_dataset, columns):
    B0 = np.append(arr = np.ones((ones_length,1)).astype(int), values=x_dataset, axis=1)
    ols = sm.OLS(endog=y_dataset, exog=x_dataset.iloc[:,columns]).fit()
    return ols.summary()

def train(x_dataset, y_dataset, test_size=.33):
    x_train, x_test, y_train, y_test = tts(x_dataset,y_dataset, test_size=test_size)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predict = lr.predict(x_test)


    result = []
    index = 0
    for y_item in y_test.values:
        predicted_item = predict[index]
        index += 1
        result.append((float(y_item), float(predicted_item)))
    
    return result, y_test.values, predict

#Smaller error is better
def rmse(p, y):
    index = 0
    total = 0  
    for p_item in p:
        y_item = y[index]
        index += 1
        total += math.pow((p_item - y_item), 2)

    return math.sqrt(total / len(y)) 
    

dataset = pd.read_csv("odev_tenis.csv")
#dataset = dataset.apply(le.fit_transform) kalan tüm kolonlara LabelEncoder.fit_transform fonksiyonunu uygular.

#dataset_without_humidity = remove_column(dataset, 2,3)
humidity = dataset.iloc[:,2]# it' s y (dependent values)


#x' s independent values
outlook = pd.DataFrame(data = categorical_to_numeric(dataset.iloc[:,0].values), columns=["sunny","overcast","rainy"])
temperature = dataset[["temperature"]]
windy = pd.DataFrame(data = le.fit_transform(dataset.iloc[:,3]), columns=["windy"])
play = pd.DataFrame(data = le.fit_transform(dataset.iloc[:,4]), columns=["play"])

final_dataset = pd.concat([outlook, temperature, windy, play],axis=1)




print("------------------------------------------------------------------------------------")
print("\n\n\n\n\n")
result, y, p = train(final_dataset, humidity)
rmse_result = rmse(p , y)
summary = ols_summary(14, final_dataset, humidity, [0,1,2,3,4,5])
print(summary)


print("------------------------------------------------------------------------------------")
print("\n\n\n\n\n")
final_dataset = remove_column(final_dataset,4,5)

result_shouldbe_better, y, p = train(final_dataset, humidity)
rmse_result_shouldbe_better = rmse(p, y)
summary = ols_summary(14, final_dataset, humidity, [0,1,2,3,4])
print(summary)