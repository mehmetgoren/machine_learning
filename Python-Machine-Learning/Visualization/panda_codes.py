"""
    panda visualtion
"""


import matplotlib.pyplot as plt
import seaborn as sb#bunu ekleyince görsellik seaborn a benziyor ve iyileşiyor
import pandas as pd

df1 = pd.read_csv("df1")
df1["A"].plot(kind="hist", bins=30) 