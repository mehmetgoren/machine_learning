import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


le = LabelEncoder()
ohe = OneHotEncoder()
ss = StandardScaler()

def remove_column(dataframe, left, right):
    left_dataframe = dataframe.iloc[:,:left]
    right_dataframe = dataframe.iloc[:,right:]
    return pd.concat([left_dataframe, right_dataframe], axis=1)


dataset = pd.read_csv("50_Startups.csv")
corr = dataset.corr()


state = dataset.iloc[:,3]
state = le.fit_transform(state)
state = ohe.fit_transform(state.reshape(-1,1)).toarray()
state = pd.DataFrame(data=state, columns=["Cl","Fl","Ny"])
state = remove_column(state,0,1)#to avoid dummy variable trap.


X = dataset.iloc[:,0:3]
X = pd.concat([X, state], axis=1)
X = ss.fit_transform(X)

y =dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = tts(X, y, test_size=.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

#şimdi de B0-bias ekleyelim.
