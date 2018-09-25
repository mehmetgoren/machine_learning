"""
    RNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 -  Data Preprocessing
dataset = pd.read_csv("usd_try.csv")
X = dataset.iloc[:,3:4].values.copy()

#scaling but this time we use normalization.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)#normilzed between zero and one.


#bu kısım t-1,t,t+1' i simile etmewk için eklendi.
step = 60
def split(X):
    #creating a data structure with 60 timesteps and 1 output.
    length = len(X)
    X_train = []
    y_train = []
    for j in range(step, length):
        X_train.append(X[j-step:j,0])
        y_train.append(X[j, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train


X_train, y_train = split(X.copy())







#building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dropout 

regressor = Sequential()

#lets add some LSTM layers.
regressor.add(CuDNNLSTM(units=250, return_sequences=True, input_shape= (X_train.shape[1], 1)))#eğer başka LSTM layer' ları ekleyteceksen bu True olmalı. units = nöronlar.
regressor.add(Dropout(0.2))

#regressor.add(CuDNNLSTM(units=200, return_sequences=True))#eğer başka LSTM layer' ları ekleyteceksen bu True olmalı.
#regressor.add(Dropout(0.2))

regressor.add(CuDNNLSTM(units=150, return_sequences=True))#eğer başka LSTM layer' ları ekleyteceksen bu True olmalı.
regressor.add(Dropout(0.2))

#regressor.add(CuDNNLSTM(units=100, return_sequences=True))#eğer başka LSTM layer' ları ekleyteceksen bu True olmalı.
#regressor.add(Dropout(0.2))

regressor.add(CuDNNLSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(CuDNNLSTM(units=25))
regressor.add(Dropout(0.2))



#adding the output layer
regressor.add(Dense(units=1))

#lets compile the RNN
regressor.compile(optimizer="adam", loss = "mean_squared_error")

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


X_test = X_train.copy()
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

real_stock_price =  dataset.iloc[:,3:4].values.copy()

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Usd-Try')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Usd-Try')
plt.title('Usd-Try Prediction')
plt.xlabel('Time')
plt.ylabel('Usd-Try')
plt.legend()
plt.show()





X_test = X_train[len(X_train)-1:len(X_train),:].copy()
#X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
y_pred = sc.inverse_transform(regressor.predict(X_test))