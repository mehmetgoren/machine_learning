"""
    by Gökalp Gören
"""
import pandas as pd
import statsmodels.formula.api as sm


def remove_column(dataframe, left, right):
    left_dataframe = dataframe.iloc[:,:left]
    right_dataframe = dataframe.iloc[:,right:]
    return pd.concat([left_dataframe, right_dataframe], axis=1)

def summary(x, y):
    elem_ols = sm.OLS(endog=y, exog=x)#ordinary least squares
    elem = elem_ols.fit()
    return elem.summary()