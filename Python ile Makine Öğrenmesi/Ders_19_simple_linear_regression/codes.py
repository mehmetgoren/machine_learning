import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("satislar.csv")

aylar = dataset[["Aylar"]] #dataset.iloc[:,0].values.reshape(-1,1)#independent variables
satislar = dataset[["Satislar"]] #dataset.iloc[:,1].values.reshape(-1,1)#dependent variables


#x' ler data, y' ler sonuçlar.
x_train, x_test, y_train, y_test = tts(aylar, satislar, test_size=.33) #1/3 olarak bölünüyor.

#sc = StandardScaler()
#standart_x_train = sc.fit_transform(x_train)
#standart_x_test = sc.fit_transform(x_test)
#standart_y_train = sc.fit_transform(y_train)
#standart_y_test = sc.fit_transform(y_test)
#
#
#lr = LinearRegression()
#lr.fit(standart_x_train, standart_y_train)#model inşası
#
#predict = lr.predict(standart_x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)#model inşası

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.title("Aylara Göre Satıi")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.plot(x_train, y_train)
plt.plot(x_test, predict)



#şimdi standart_x_test ile predict' i karşılaştırabiliriz.