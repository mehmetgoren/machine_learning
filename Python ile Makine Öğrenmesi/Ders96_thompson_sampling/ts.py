"""
    Thompson Sampling
"""

import pandas as pd
import matplotlib.pyplot as plt

import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")



row_length = dataset.shape[0]
col_length = dataset.shape[1]
total = 0
selectedList = []
ones = [0] * col_length
zeroes = [0] * col_length


selectedList = []
total = 0
for n in range(1, row_length):
    ad = 0#selected ad
    max_th = 0
    for i in range(0, col_length):
        random_beta = random.betavariate(ones[i] + 1, zeroes[i] + 1)#this is the real algorithm(Thompson Sampling)

        if random_beta > max_th:
            max_th = random_beta
            ad = i
            
        selectedList.append(ad)
        trophy = dataset.values[n, ad]
        if trophy == 1:
            ones[ad] += 1
        else:
            zeroes[ad] += 1
            
        total += trophy
 
print("Total: ", total)       
        
plt.hist(selectedList)
plt.title("Thompson Sampling")
plt.show()# 4 numaralı reklam en çok tıklanan
        
    