"""
    decision tree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Position_Salaries.csv")

corr = dataset.corr()

def plot(independent, dependent, predict):
    plt.scatter(independent,dependent, color="red")
    plt.plot(independent, predict, color="blue")
    plt.xlabel("Level")
    plt.ylabel("Salary")
    plt.show()
    
def plot2(independent, dependent, regressor):
    # Visualising the Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(independent), max(independent), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(independent, dependent, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Decision Treen Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#from sklearn.preprocessing import StandardScaler. Scaler a gerek yok
#scaler_X = StandardScaler()
#X = scaler_X.fit_transform(X)
#
#
#scaler_y = StandardScaler()
#y = scaler_y.fit_transform(y.values.reshape(-1,1))


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


y_pred=regressor.predict(6.5)

#plot(X,y, y_pred)
plot2(X,y,regressor)