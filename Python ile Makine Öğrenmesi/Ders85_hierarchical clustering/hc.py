"""
    hierarchical clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans


def show(x, y, title):
#    for j in range(cluster_count):
#        plt.scatter(x[y==j,0], x[y==0,1], s=100, c=color)
    plt.scatter(x[y==0,0], x[y==0,1], s=100, c="red")
    plt.scatter(x[y==1,0], x[y==1,1], s=100, c="green")
    plt.scatter(x[y==2,0], x[y==2,1], s=100, c="blue")
    plt.scatter(x[y==3,0], x[y==3,1], s=100, c="yellow")
    plt.title(title)
    plt.show()

dataset = pd.read_csv("musteriler.csv")

x = dataset.iloc[:,2:4].values#yaş ve hacimi aldık.

ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
y = ac.fit_predict(x)# dönen 0,1,2 sonuçları hangi cluster' a (Kümeye) ait olduğunu gösteriyor.


#böylece 4 cluster' ı da fgörüyoruz
print("HC")
show(x,y,"HC")




print("\n\n\n")
#lets compare hc and kmeans.
km = KMeans(n_clusters=4, init="k-means++", random_state=123)
y = km.fit_predict(x)
print("kmeans")
show(x,y, "kmeans")


print("\n\n\n\n\n\n\n\n")
#bir de dendogram a bakalım
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method="ward"))#asalında 3 almak mantıklı değilmiş. 2 ve 4 teki kırılmalara yakın bir sonuç elde ettik. Yani 3' ü kullanmamak gerek.
plt.show()