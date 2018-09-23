"""
    Hierarchical Clustering
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Mall_Customers.csv")#spending score var. bir classification yok
#unsupervised bir şekilde segmentlere ayır müşterileri

X = dataset.iloc[:,[3,4]].values#yıllık gelir ve score


#dendogram dan cluster sayısını bulallım
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Öklit Uuzunluğu")
plt.show()


#şimdi de dataset' e uygulayalım. n_cluster=5 dendogram dan egliyorr
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)



#visualising it, bitch...
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c="red",label="Varyemezler")#Cluster1, çok para var az harcıyot
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c="blue",label="Standart")#Cluster, ortalama
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c="green",label="Zengin ve çok harcayan")#Cluster3, çok para var çok harcıyor
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c="cyan",label="Fakir Ama Çok Harcayan")#Cluster4, az para var çok harcıyor
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c="magenta",label="Fakirler")#Cluster5, az para var az harcıyor

#plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1], s=300, c="yellow", label="Centroids")
plt.title("Cluıster of Clients")
plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

