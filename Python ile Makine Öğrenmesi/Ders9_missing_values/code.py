# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:46:55 2018

@author: mehme
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer as Imp #sci-kit learn bilimsel makina öğrenmesi imputer == töhmet, buradaki affetmek

missing_values = pd.read_csv("missing_values.csv")

imputer = Imp(missing_values="NaN", strategy="mean", axis=0) # this is a strategy. Nan for null values, mean for avarages of numbers

yas = missing_values.iloc[:,1:4].values#1 den 4 e kadar değerleri al ki bu verisetinde bunlar sayısal verilerdir.

print(yas)
print("-----------------------------------------")
print("\n")

imputer = imputer.fit(yas[:,1:4])

clean_values= imputer.transform(yas[:,1:4])

print(clean_values)# in this dataset, NaN and mean_value has been replaced so we have an new dataset that does not cause a data noise.