"""
    Unsupervised K-Means
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



df = pd.read_csv("College_Data", index_col=0)

#sb.lmplot(x="Room.Board", y="Grad.Rate", data=df,hue="Private", fit_reg=False, size=6,aspect=1)
#plt.show()
#
#sb.lmplot(x="Outstate", y="F.Undergrad", data=df,hue="Private", fit_reg=False, size=6,aspect=1)
#plt.show()


X = df.drop("Private", axis=1)

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=2)
cluster.fit(X)

cluster_center = cluster.cluster_centers_