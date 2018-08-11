"""
    XGBoost - ANN
"""



import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split as tts


dataset= pd.read_csv("Churn_Modelling.csv")

geo = dataset.iloc[:,4]
le = LabelEncoder()
geo = le.fit_transform(geo)

ohe = OneHotEncoder()
geo = ohe.fit_transform(geo.reshape(-1,1)).toarray();
geo = pd.DataFrame(data=geo, columns=["fr","gr","sp"])

sex = dataset.iloc[:,5]
sex = le.fit_transform(sex)
sex = pd.DataFrame(data=sex, columns=["sex"])

credit_score = dataset.iloc[:,3]

x = dataset.iloc[:,6:-1]

x = pd.concat([credit_score,geo,sex,x], axis=1)

y = dataset.iloc[:,-1]


x_train, x_test, y_train, y_test = tts(x,y,test_size=.2,random_state = 0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)



from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)