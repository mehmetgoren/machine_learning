import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("Data.csv")


#ülkeleri numerik yaptık.
le = LabelEncoder()
countries = dataset.iloc[:,0].values
countries = le.fit_transform(countries)
ohe = OneHotEncoder()
countries = ohe.fit_transform(countries.reshape(-1,1)).toarray()


#x ve miisng values i mean ile düzelttik
x = dataset.iloc[:,1:3].values
imputer = Imputer()
x = imputer.fit_transform(x)

y = le.fit_transform(dataset.iloc[:,-1].values)

countries = pd.DataFrame(data=countries, columns=["Fra","Ger","Spa"])
x = pd.DataFrame(data=x, columns=["Age","Salary"])
x = pd.concat([countries, x],axis=1)

y = pd.DataFrame(data=y, columns=["Purchased"])

x_train, x_test, y_train, y_test = tts(x, y, test_size=.2, random_state=0)


#maaş ve yaş arasında ciddi fark var ve bu öklit uzunlupğunda yaşın yutulmasınıa sebeb olur. O yüzden scale edelim
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
