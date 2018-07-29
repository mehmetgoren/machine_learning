import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer #affedici 

dataset = pd.read_csv("missing_values.csv")
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

numeric_values = dataset.iloc[:,1:4].values
imputer = imputer.fit(numeric_values)

clean_values = imputer.transform(numeric_values)