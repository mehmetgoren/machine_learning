"""
    Apriori
"""

import pandas as pd
import matplotlib.pyplot as plt

from lib.apyori import apriori

dataset = pd.read_csv("sepet.csv", header=None)

t = [];
r_length = dataset.shape[0]
c_length = dataset.shape[1]
for row in range(r_length):
    item = []
    for col in range(c_length):
        item.append(str(dataset.iloc[row, col]))
    t.append(item)
    
rules = apriori(t, min_support=0.01, min_confidence=.2, min_lift=3, min_length=2)

result = list(rules)
#result = json.loads(result)

print(result)