"""
    PCA
"""


import numpy as np
import pandas as pd


def confusion_matrix(y, p):
    matrix = conf_m(y, p)
    return matrix

dataset = pd.read_csv("Wine.csv")
corr = dataset.corr()#koreleasyon matrix' ine göre cinsiyet anlamsız zaten.



X = dataset.iloc[:,0:-1]

y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=0)


from sklearn.preprocessing import StandardScaler
ss  =StandardScaler() 
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)




#appliying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
X_train = pca.fit_transform(X_train)#!!!! çok önemli PCA unsupervised olduğu için y(Dependent)' yi, vermiyoruz. (k-means) gibi varyansı mazimize ediyor.
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)#çıkan değerler sürekli değişmesin diye.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import  confusion_matrix as conf_m
cm = confusion_matrix(y_test, y_pred)



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot(x, y):
    x1, x2 = np.meshgrid(np.arange(start = x[:, 0].min() - 1, stop = x[:, 0].max() + 1, step = 0.01),
                         np.arange(start = x[:, 1].min() - 1, stop = x[:, 1].max() + 1, step = 0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(x[y == j, 0], x[y == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    plt.title('logistic regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
   

plot(X_train, y_train)
plot(X_test, y_test)
