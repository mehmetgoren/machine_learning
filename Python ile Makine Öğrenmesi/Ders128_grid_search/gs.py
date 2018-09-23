"""
    Grid Search
    Bu Dest model seçimi için en önemli kısım.
"""

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

x_train, x_test, y_train, y_test = tts(x, y, test_size=.25, random_state = 0)#random_state= 0 yap her zaman.


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)




#şimdi de k-fold ile split edelim

from sklearn.model_selection import cross_val_score
#estimater: classfier burada
#cv = kaç katlamalı. Ayrıntılar Ders 127' de.
cvs=cross_val_score(estimator=classifier, X=x_train, y=y_train,cv=4)
mean = cvs.mean()
standart_sapma = cvs.std()


#parametre optimizasyonu ve algoritma seçimi
p = [{"C":[1,2,3,4,5], "kernel":["linear"]}, {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[1,.5,.1,.01,.001]}]#bunlar '{}' dictionary
from sklearn.model_selection import GridSearchCV
"""
    estimator=neyi optimize etmewk istediğimiz classifier algoritması
    param_grid=denenecek parametreler
    scorring=neye göre skorlanacağı örneğin accuricy, purity vs...
    cv=kaç kalmalaı 1/4,2/4,3/4
    n_jobs=aynı anda çalışacak iş.
"""
gs = GridSearchCV(estimator=classifier,param_grid=p,scoring="accuracy", cv = 10)
grid_search = gs.fit(x_train, y_train)#bu bişr proxy çağrısı, svn' i, çağırıyor.
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_score)
print(best_parameters)