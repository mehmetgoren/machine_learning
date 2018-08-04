"""
    k-means
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



dataset = pd.read_csv("musteriler.csv")

x = dataset.iloc[:,2:4]#yaş ve hacimi aldık.




km = KMeans(n_clusters=3, init="k-means++")
km.fit(x)

cluster_centers = km.cluster_centers_

results = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init="k-means++", random_state=123)#random state' ın amacı sabit bif ifadeden farklı kümeler oluştur ki, daha iyi değerlendirelim. random bu ölçüyü bozar.
    km.fit(x)
    results.append(km.inertia_)#bu inertia wcss değerleri. (Yani k-means in ne kadar başarılı olduğu)
    
    
plt.plot(range(1,11), results, color="red")#bu sonuca göre kırılma noktası n_clusters 2, 3, 4 olabilir. eğri kırılmaları buralarda.
#yani en optimum n_clusters sayısı buradaki grafiğe göre 2,3,4 ki bence 4 olacaktır.
