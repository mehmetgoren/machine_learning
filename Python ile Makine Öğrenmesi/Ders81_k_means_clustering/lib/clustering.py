"""
    k-means ny Gökalp Gören
"""

from sklearn.cluster import KMeans

def k_means(x, n_clusters=3, init="k-means++"):
    km = KMeans(n_clusters=n_clusters, init=init)
    km.fit(x)
    return km
