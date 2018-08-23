"""
    backward elemination
"""

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
X = pd.concat([state, X], axis=1)
#X = ss.fit_transform(X)
#
#y =dataset.iloc[:,-1].values
#
#X_train, X_test, y_train, y_test = tts(X, y, test_size=.2, random_state=0)
#
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#y_pred=regressor.predict(X_test)
#
##şimdi de B0-bias ekleyelim.
bias = np.ones((50,1)).astype(int)
bias = pd.DataFrame(data=bias, columns=["bias"])
X = pd.concat([bias, X], axis=1)
y = dataset.iloc[:,-1]
#X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)#♣ilk kolona bias ekledik.

#X_opt = X[:,[0,1,2,3,4,5]]
#
#lets calculate the p-value
X_opt = X.iloc[:,[0,1,2,3,4,5]]
import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
#
print("----------------------------------------------------------------")
print("\n\n\n\n")

X_opt = X.iloc[:,[0,3]]#burası kolonları atıyor.....
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
summary = regressor_OLS.summary()
print(summary)



#def backwardElimination(x, sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)


#import statsmodels.formula.api as sm
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int)
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
#                        print (regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)