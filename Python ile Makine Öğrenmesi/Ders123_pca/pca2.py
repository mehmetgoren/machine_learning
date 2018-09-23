"""
    PCA Boyut indirgeme
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv("Wine.csv")
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values



ss = StandardScaler()
x = ss.fit_transform(x)



#pca starts here.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)#iki boyuta işndirge dedik.
x_pca = pca.fit_transform(x)#uhaa 13 boyutu 2 boyuta indirgendi.



#şimdi ML uygulama zamanı ki PCA' ın indirgediği boyutları karşılaştıralım. ne kadar başarılı acaba?
from sklearn.linear_model import LogisticRegression

#pca dönüşümünden önce gelen
classifier = LogisticRegression(random_state=0);#random_state == sabit değerlşe logistic regressiopn kullan
classifier.fit(x, y)
y_pred = classifier.predict(x)


#pca dönüşümünden sonra gelen
classifier_pca = LogisticRegression(random_state=0);#random_state == sabit değerlşe logistic regressiopn kullan
classifier_pca.fit(x_pca, y)
y_pred_pca = classifier_pca.predict(x_pca)


#bir de karşılaştıralım
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred) 

cm_pca = confusion_matrix(y, y_pred_pca)


#13 kolondan 2 ye düşürdük ancak confusion matrix bize yine de çok iyi sonuçlar verdiğini söylüyor..
# bu arada wine örneği logistic_regression a tam fit oluyor.
#pca farklı durumlarda başarıyı artııtrabilir hatta.