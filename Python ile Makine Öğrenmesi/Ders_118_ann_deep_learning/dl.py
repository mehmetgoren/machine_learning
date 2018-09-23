"""
    deep learninhg with tensorflow - keras
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split as tts


dataset= pd.read_csv("Churn_Modelling.csv")

le = LabelEncoder()

x = dataset.iloc[:,3:-1].values
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
#x = x[:,1:]
#print(x)

y = dataset.iloc[:,-1].values
#
#
x_train, x_test, y_train, y_test = tts(x,y,test_size=.2)
#
#
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


#lets do some Artificial Neural Network
import keras
from keras.models import Sequential #yani ANN' yi kullanacağım
from keras.layers import Dense #yani nöronları oluşturacağımız nesne
#from keras.optimizers

#yapay sinir ağını sınıflandırma için kullanıyoruz.

classifier = Sequential() #artık bizim de bir yuapaz sinir ağımız var

#giriş katmanı. input katmanı olduğu için input_dim' i belirtmek zorundayız.
classifier.add(Dense(13, init="uniform", activation="relu", input_dim=12))#katman ekledik. uniform' da dağılım sen bizim ann' deki sinapsisleri üzerine verileri initialize et.
#sonraki süreçte ann bu parametreleri optimize edecek.

#activation=relu de aktivasyon fonksiyonunu rectifier olmasını istedik.


#birtane daha gizli katman ekleyelim
#ikinci bir hidden layer ekleyelim.
classifier.add(Dense(17, init="uniform", activation="relu"))




#•çıkış/y/bağımlı katman
classifier.add(Dense(1, init="uniform", activation="sigmoid"))#çıkış katmanı linear değil logoritmik, sigmoid olması adetdir.


#artık derleyelim (tf.session.run?)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])#adam daha gelişmiş scothastic Gradient Descent versiyonudır.

#optimizer parametereleri optimize edecek ancak loss function bu değerlendirmeyi yapacak fonksiyondur. (byrada binary kullanıyoruz)


#şimdi predict zamanı
classifier.fit(x_train, y_train, epochs=50)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > .5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)