"""
    PCA Boyut indirgeme
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split as tts


dataset = pd.read_csv("Wine.csv")
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values


x_train, x_test, y_train, y_test = tts(x,y,test_size=.2)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


#pca starts here.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)#iki boyuta işndirge dedik.
x_train2 = pca.fit_transform(x_train)#uhaa 13 boyutu 2 boyuta indirgendi.
x_test2 = pca.fit_transform(x_test)


#şimdi ML uygulama zamanı ki PCA' ın indirgediği boyutları karşılaştıralım. ne kadar başarılı acaba?
from sklearn.linear_model import LogisticRegression

#pca dönüşümünden önce gelen
classifier = LogisticRegression(random_state=0);#random_state == sabit değerlşe logistic regressiopn kullan
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


#pca dönüşümünden sonra gelen
classifier2 = LogisticRegression(random_state=0);#random_state == sabit değerlşe logistic regressiopn kullan
classifier2.fit(x_train2, y_train)
y_pred2 = classifier2.predict(x_test2)


#bir de karşılaştıralım
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm2 = confusion_matrix(y_test, y_pred2)