import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder 

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


sex = dataset.iloc[:,-1].values

sex_table = pd.DataFrame(data=sex, columns=["cinsiyet"])

result = pd.concat([datatable, datatable2], axis=1)#axis=1 diyerek satır bazlı değil, kolon bazlı değiştir diyoruz.
result_with_cinsiyet = pd.concat([result, sex_table], axis=1)