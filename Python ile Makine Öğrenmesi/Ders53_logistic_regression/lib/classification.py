"""
    by Gökalp Gören
"""

from sklearn.linear_model import LogisticRegression
import scaler


def logistic_regression(x, y, scaleParameters= True)
    if scaleParameters:
        x, y = scaler.standart_scaler(x, y)
    lr = LogisticRegression()
    lr.fit(x, y)#train them
    return lr.predict(x)