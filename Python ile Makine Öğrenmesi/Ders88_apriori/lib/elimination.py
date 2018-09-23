"""
   Elemination functions by Gökalp Gören
"""
import pandas as pd


def remove_column(dataframe, left, right):
    left_dataframe = dataframe.iloc[:,:left]
    right_dataframe = dataframe.iloc[:,right:]
    return pd.concat([left_dataframe, right_dataframe], axis=1)