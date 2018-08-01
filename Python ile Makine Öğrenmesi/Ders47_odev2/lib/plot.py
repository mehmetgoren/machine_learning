"""
    by Gökalp Gören
"""

import matplotlib.pyplot as plt

def draw(x, y, predict):
    plt.scatter(x, y, color="red")
    plt.plot(x, predict, color="blue")
    plt.show()#refresh eder.