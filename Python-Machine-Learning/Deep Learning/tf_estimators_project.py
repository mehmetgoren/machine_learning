"""
    Tensorflowe Estimator Project
"""

import pandas as pd 

df = pd.read_csv("bank_note_data.csv")


import matplotlib.pyplot as plt
import seaborn as sb

#sb.countplot(x="Class" , data=df)
#plt.show()
#
#sb.pairplot(df, hue="Class")#görüldüğü üzere veri oldukça ayrılabilir.
#plt.show()


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(df.drop("Class", axis=1))
y = df.iloc[:,-1].values


#gerisi öçnceki estimator' ın aynısı.