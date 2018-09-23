"""
    2nd homework
"""

import pandas as pd
from sklearn.metrics import r2_score
from lib import algorithms as a 
from lib import plot, scaler, elimination


dataset = pd.read_csv("maaslar_yeni.csv")


x = dataset.iloc[:,2:5]
y = dataset.iloc[:,5]





#linear regression
lr = a.linear_regression(x, y)
#lr_score = r2_score(y, lr)
summary = elimination.summary(x, lr)
print(summary)

print("\n")
print("---------------------------------------------------------------------------------------------")
print("\n")

#şimdi p-value' sı büyükmlanları çıkaralım
#removed_x = elimination.remove_column(x,1,3)
#lr = a.linear_regression(removed_x, y)
#summary = elimination.summary(removed_x, lr)
#print(summary)
#



pr = a.polynomial_regression(x, y)
summary = elimination.summary(x, pr)
print(summary)
#pr_score = r2_score(y, pr)

#x_scaled, y_scaled = scaler.standart_scaler(x.iloc[:,-1].values.reshape(-1,1), y.values.reshape(-1,1))
#svr = a.svr_predict(x_scaled, y_scaled)
#svr_score=r2_score(y_scaled, svr)


dt = a.decision_tree(x, y)
dt_score = r2_score(y, dt)

rf = a.random_forest_regression(x, y)
rf_Score = r2_score(y, rf)


print("\n")
print("---------------------------------------------------------------------------------------------")
print("\n")
print("Korelasyon Matrix' i")

corr = dataset.corr()#köşegenm simetrik matrix
#aslında sadece buna bakarak ta çok iyi bir elemination yapılabilir.