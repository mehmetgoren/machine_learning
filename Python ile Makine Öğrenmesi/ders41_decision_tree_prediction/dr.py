"""
    decision tree
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

dataset= pd.read_csv("maaslar.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(x, y)#train, x ve y arasındaki lişkiyi öğrenmesini istiyoruz.
predict = tree_regressor.predict(x)#birebir öğrendi çünü her zaman ' ye böldü ve bu sonuç üzerinden gider


plt.scatter(x, y, color="red")
plt.plot(x, predict, color="blue")
plt.show()#refresh eder.


#overfitting here it comes
z = x + .5
k = x - .5
plt.plot(x, tree_regressor.predict(z), color="yellow")
plt.plot(x, tree_regressor.predict(k), color="green")
#

print(tree_regressor.predict(11))#overfitting!!!!!
print(tree_regressor.predict(6.6))#overfitting!!!!!!

#print("\n\n\n\n\n\n\n")

#sonuçlar her zaman y ne ise ona göre çıkacaktır. yani, bu algoritmaya prediction için kullanmak hiç de iyi bir fikir değildir.