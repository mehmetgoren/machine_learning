"""
    Unsupervised Deep Learning -> SElf Organizing Map
"""
import numpy as np
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)


#lets some som :D

print(X.shape[0])
print(X.shape[1])

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.2)#X=10, y=10 demek oluşacak grid' in birimleri
som.random_weights_init(X)
som.train_random(data=X, num_iteration=512)

print(som.distance_map().T)
#lets visualize
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o","s"]
colors = ["r","g"]
for index, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5
         , markers[y[index]]
         , markeredgecolor=colors[y[index]]
         , markerfacecolor="None"
         , markersize = 10
         , markeredgewidth = 2)
show()


mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,8)], mappings[(6,4)]), axis = 0)
frauds = sc.inverse_transform(frauds)