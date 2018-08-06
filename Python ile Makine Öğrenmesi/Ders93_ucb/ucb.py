"""
    UCB - Upper Confidence Bound
"""

import math
import pandas as pd
import matplotlib.pyplot as plt

import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#this is the most dumb algorithm in the world (Random Selection)
row_length = dataset.shape[0]
col_length = dataset.shape[1]
total = 0
selectedList = []
for j in range(row_length):
    ad= random.randrange(col_length)
    selectedList.append(ad)
    trophy = dataset.values[j, ad]
    total += trophy
    
    
plt.hist(selectedList)
plt.title("random")
plt.show()


#lets write a more sophisticated algorithm (ucb)
#Ri(n)
trophies = [0] * col_length

#Ni(n)
clicks = [0] * col_length

selectedList = []
total = 0
for n in range(1, row_length):
    ad = 0#selected ad
    max_ucb = 0
    for i in range(0, col_length):
        if (clicks[i] > 0):
            mean = trophies[i] / clicks[i]
            delta = math.sqrt(3/2*math.log(n)/clicks[i])
            ucb = mean+delta
        else:
            ucb = row_length * 10
        
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
            
        selectedList.append(ad)
        clicks[ad] += 1
        trophy = dataset.values[n, ad]
        trophies[ad] += trophy
        total += trophy
 
print("Total: ", total)       
        
plt.hist(selectedList)
plt.title("UCB")
plt.show()# 4 numaralı reklam en çok tıklanan
        
    