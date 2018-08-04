"""
   scaling functions  by Gökalp Gören
"""

from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split



def encode_label(column):
    le = LabelEncoder()
    column_as_numbers = le.fit_transform(column)
    return column_as_numbers


def encode_onehot(columns, categorical_features="all"):
    ohe = OneHotEncoder(categorical_features=categorical_features)
    columns_as_matrix = ohe.fit_transform(columns.reshape(-1,1)).toarray()#reshape işte ger bir item' ı array e çeviriyor.
    return columns_as_matrix



def standart_scaler(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)



def impute(dataset):
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    return imputer.fit_transform(dataset)



def split(x, y, test_size=.33):
    return train_test_split(x, y, test_size=test_size)