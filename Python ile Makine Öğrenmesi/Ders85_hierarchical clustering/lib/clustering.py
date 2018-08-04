"""
    k-means ny Gökalp Gören
"""

from sklearn.cluster import KMeans, AgglomerativeClustering

def k_means(x, n_clusters=3, init="k-means++"):
    km = KMeans(n_clusters=n_clusters, init=init)
    km.fit(x)
    return km


def hierarchicalclustering(x, n_clusters=2, affinity="euclidean", linkage="ward"):
    ac = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    y = ac.fit_predict(x)# dönen 0,1,2 sonuçları hangi cluster' a (Kümeye) ait olduğunu gösteriyor.
    return y;

