"""
    LDA Boyut indirgeme
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


#şimdi sıra LDA' de
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train, y_train)#y_train i vermemmizin sebebi sınıfları belirlemek (supervised)
x_test_lda = lda.transform(x_test)


#şimdi ML uygulama zamanı ki PCA' ın indirgediği boyutları karşılaştıralım. ne kadar başarılı acaba?
from sklearn.linear_model import LogisticRegression

#pca dönüşümünden önce gelen
classifier_lda = LogisticRegression(random_state=0);#random_state == sabit değerlşe logistic regressiopn kullan
classifier_lda.fit(x_train, y_train)
y_pred_lda = classifier_lda.predict(x_test)



#bir de karşılaştıralım
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_lda)