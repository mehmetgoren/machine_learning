import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder 

dataset = pd.read_csv("missing_values.csv")

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

clean_values = imputer.fit_transform(dataset.iloc[:,1:4].values)


countries = dataset.iloc[:,0].values
#countries = countries.reshape(-1, 1)
print(countries)

le = LabelEncoder()
countries_as_numbers = le.fit_transform(countries)

#temp = np.array(countries_as_numbers, copy=True)
#countries_as_numbers = []
#for i in temp:
#    countries_as_numbers.append([i])
#    
#countries_as_numbers = np.array(countries_as_numbers)
#print("\n\n\n")
#print(countries_as_numbers)

#countries_as_numbers[2]=0
#print(countries_as_numbers[2])
#print(temp[2])

#print(countries_as_numbers)

ohe = OneHotEncoder(categorical_features="all")

countries_as_matrix = ohe.fit_transform(countries_as_numbers.reshape(-1,1)).toarray()#reshape işte ger bir item' ı array e çeviriyor.
print("\n\n\n")
print(countries_as_matrix)
