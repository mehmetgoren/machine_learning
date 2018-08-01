"""
    by Gökalp Gören
"""

from sklearn.preprocessing import StandardScaler

def standart_scaler(x, y):    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)
    return x_scaled, y_scaled