import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder 
from sklearn.cross_validation import train_test_split as tts


dataset = pd.read_csv("missing_values.csv")

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

clean_values = imputer.fit_transform(dataset.iloc[:,1:4].values)


countries = dataset.iloc[:,0].values
#print(countries)

le = LabelEncoder()
countries_as_numbers = le.fit_transform(countries)



ohe = OneHotEncoder(categorical_features="all")
countries_as_matrix = ohe.fit_transform(countries_as_numbers.reshape(-1,1)).toarray()#reshape işte ger bir item' ı array e çeviriyor.


datatable = pd.DataFrame(data=countries_as_matrix, columns=["fr","tr","us"])
datatable2 = pd.DataFrame(data=clean_values, columns=["boy","kilo","yas"])
clean_dataset = pd.concat([datatable, datatable2], axis=1)#axis=1 diyerek satır bazlı değil, kolon bazlı değiştir diyoruz.


sex = dataset.iloc[:,-1].values
sex_table = pd.DataFrame(data=sex, columns=["cinsiyet"])

 
#x' ler data, y' ler sonuçlar.
x_train, x_test, y_train, y_test = tts(clean_dataset, sex_table, test_size=.33) #1/3 olarak bölünüyor.

