"""
   scaling functions  by Gökalp Gören
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split




def standart_scaler(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)



def impute(dataset):
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    return imputer.fit_transform(dataset)



def split(x, y, test_size=.33):
    return train_test_split(x, y, test_size=test_size)