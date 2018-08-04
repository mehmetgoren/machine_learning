"""
  ecaluation functions by Gökalp Gören
"""

import statsmodels.formula.api as sm
from sklearn.metrics import r2_score, confusion_matrix as cm, roc_curve 

#regression
def r2(y, p):
	return r2_score(y, p)

def summary(x, y):
    elem_ols = sm.OLS(endog=y, exog=x)#ordinary least squares
    elem = elem_ols.fit()
    return elem.summary()

def corr(dataframe):
    return dataframe.corr()
#regression


 #classification
def confusion_matrix(y, p):
    matrix = cm(y, p)
    return matrix
#classification
        

#bU roc a daha sıkı çalışmamız gerekecek.
#ROC, TPR, FPR
#receiver Operating Chjaracterisdtic  y_proba probabality oranlarını sayısal olarak verir.
def roc(y, y_proba, pos_label="yes"):#y_proba predict_proba fonksiyonundan geliyor.
    fpr, tpr, threshold = roc_curve(y, y_proba, pos_label)#false positive rate, true positive rate
    return fpr, tpr, threshold