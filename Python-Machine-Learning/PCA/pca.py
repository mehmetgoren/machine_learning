"""
    PCA - Unsupervised.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()


df = pd.DataFrame(data["data"], columns=[data["feature_names"]])
X = df.iloc[:,:].values
y = data["target"]#unnecessery


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)  



from sklearn.decomposition import PCA
pca = PCA(n_components=2)#2 boyuta indirge
X_pca = pca.fit_transform(X)


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=data["target"],cmap="plasma")
plt.xlabel("First PCA")
plt.ylabel("Second PCA")
plt.show()

components = pca.components_

df_comp = pd.DataFrame(components, columns=[data["feature_names"]])

plt.figure(figsize=(12,6))
sb.heatmap(df_comp, cmap="plasma")
plt.show()