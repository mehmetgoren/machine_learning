"""
   SVM
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import  confusion_matrix as conf_m

def confusion_matrix(y, p):
    matrix = conf_m(y, p)
    return matrix



dataset = pd.read_csv("Social_Network_Ads.csv")
corr = dataset.corr()#koreleasyon matrix' ine göre cinsiyet anlamsız zaten.


X = dataset.iloc[:,2:4]
y = dataset.iloc[:,-1]


X_train, X_test, y_train, y_test = tts(X,y,test_size=.2, random_state=0)
ss  =StandardScaler() 
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=0)#çıkan değerler sürekli değişmesin diye.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


def plot(X, y, title):
    from matplotlib.colors import ListedColormap
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    

plot(X_train, y_train, 'Support Vector Machine (Training set)')
plot(X_test, y_test, 'Support Vector Machine  (Test set)')
