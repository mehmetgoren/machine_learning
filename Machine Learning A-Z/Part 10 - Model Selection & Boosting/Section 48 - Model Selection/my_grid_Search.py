"""
  grid search (optimising the hyper paramaters)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import  confusion_matrix


dataset = pd.read_csv("Social_Network_Ads.csv")
corr = dataset.corr()#koreleasyon matrix' ine göre cinsiyet anlamsız zaten.


X = dataset.iloc[:,2:4]
y = dataset.iloc[:,-1]


X_train, X_test, y_train, y_test = tts(X,y,test_size=.2, random_state=0)


ss  =StandardScaler() 
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)#çıkan değerler sürekli değişmesin diye.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


#apliying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y = y_train, cv = 10)
mean = accuracies.mean()#çok iyi.
stddev = accuracies.std();#az iyi


#apliying grid search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{"C":[1,10,100,1000], "kernel":["linear"]},
              {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.1,0.2,0.3,0.4,0.5,0.1,0.01,0.001]}
              ]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=1)#◙bu 10 cross_val_score' dan geldi.
grid_search = grid_search.fit(X_train, y_train)
best_accuracy=grid_search.best_score_
best_parameters = grid_search.best_params_#harika






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
