"""
    random forest 
"""
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def rmse(y, p):
    index = 0
    total = 0  
    for p_item in p:
        y_item = y[index]
        index += 1
        total += math.pow((p_item - y_item), 2)

    return math.sqrt(total / len(y)) 


dataset= pd.read_csv("maaslar.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)#n_estimators kaç tane decision tree çizileceği.
rf_reg.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
predict = rf_reg.predict(x)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider


plt.scatter(x, y, color="red")
plt.plot(x, predict, color="blue")
#plt.show()#refresh eder.

print("\n\n\n\n\n\n\n")
print("RMSE: ", rmse(y, predict))


print(rf_reg.predict(11))#çok daha iyi ve overfitting' den daha iyi bir sonuç
print(rf_reg.predict(6.6))

z = x+.5
plt.plot(x, rf_reg.predict(z), color="green")#artık overfitting yok.
k = x-.5
plt.plot(x, rf_reg.predict(k), color="yellow")

