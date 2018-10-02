"""
    matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,5,11)
y = x ** 2


#fonksiyonel
plt.subplot(1,2,1)#specifying the numbers of rows and columns
plt.plot(x,y,"r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("this is my first time")
plt.show()




#oop
fig = plt.figure()
axes = fig.add_axes([.1,.1,.8,.8])
axes.plot(x,y)
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_title("title")


#lets div in
fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])#.8 li
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])#.75 li. BUnlarla oyna ve yeri değişsin.

axes1.plot(x,y,"r")
axes1.set_title("LARGER PLOT")


axes2.plot(x,y,"b")
axes2.set_title("smaller plot")




fig, axes = plt.subplots(nrows=1, ncols=2)
index = 0
for current_ax in axes:
    index +=1
    current_ax.set_title(index)
    current_ax.plot(x,y)#plot fonksiyon eğrisini çiziyor.


fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
fig.savefig("my_image.png", dpi=200)



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y, color="purple", linewidth=20, alpha=.5)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y, color="purple", linewidth=3, linestyle=":")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y, color="purple", linewidth=3, marker="o", markersize=10, markerfacecolor="orange", markeredgecolor="g")



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y, color="purple", linewidth=3, marker="o", markersize=10, markerfacecolor="orange", markeredgecolor="g")
ax.set_xlim([0,1])#its a zoom
ax.set_ylim([0,2])


#scatter histogram da destekliyor.