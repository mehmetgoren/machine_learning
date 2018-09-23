"""
    K-Means
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Mall_Customers.csv")#spending score var. bir classification yok
#unsupervised bir şekilde segmentlere ayır müşterileri

X = dataset.iloc[:,[3,4]].values#yıllık gelir ve score


#using the wlbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss=[]
for j in range(1,11):
    kmeans = KMeans(n_clusters=j,init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()#5 iyi gibi gözüküyor.


#hadi k-means i uygulayalım.
kmeans=KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#buradan gelen vektör dataset observation' larının hangi kümneye ait olduğunu belirtiyor.

#şimdi bunu görsellştirelim.
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c="red",label="Varyemezler")#Cluster1, çok para var az harcıyot
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c="blue",label="Standart")#Cluster, ortalama
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c="green",label="Zengin ve çok harcayan")#Cluster3, çok para var çok harcıyor
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c="cyan",label="Fakir Ama Çok Harcayan")#Cluster4, az para var çok harcıyor
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c="magenta",label="Fakirler")#Cluster5, az para var az harcıyor

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c="yellow", label="Centroids")
plt.title("Cluıster of Clients")
plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

#Bu görseli çok feature olursa pca ile ikiyue indir ve kullanabilirsin.