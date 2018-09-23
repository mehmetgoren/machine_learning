"""
    RNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 -  Data Preprocessing
dataset_train= pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

#scaling but this time we use normalization.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)#normilzed between zero and one.

#creating a data structure with 60 timesteps and 1 output.
length = len(training_set_scaled)
X_train = []
y_train = []

#bu kısım t-1,t,t+1' i simile etmewk için eklendi.
step = 60
for j in range(step, length):
    X_train.append(training_set_scaled[j-step:j,0])
    y_train.append(training_set_scaled[j,0])
X_train, y_train = np.array(X_train), np.array(y_train)  


print(X_train.shape[0])
print(X_train.shape[1])


#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



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



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - step:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(step, step+20):
    X_test.append(inputs[i-step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()