# -*- coding: utf-8 -*-
"""
Ders 6
Kütiphanelerin Yüklenmesi 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yukleme

datas = pd.read_csv("datas.csv")
print(datas)
print(datas[["boy", "kilo"]])

# 