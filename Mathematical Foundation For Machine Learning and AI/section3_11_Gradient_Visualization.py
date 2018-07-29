import numpy as np
import matplotlib.pyplot as plt


#genereate 2D mashgrid
nx, ny = (100, 100)

x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)

xv, yv = np.meshgrid(x, y)

# define a function to plot
def f(x, y):
    return x * (y**2)#f(x) = x.y^2

#calculate Z value for each x, y point
z = f(xv, yv)

#make a color pilot to display the data
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.title('2D Color Plot of f(x,y)=xy^2')
plt.colorbar()
plt.show()


# generate 2D meshgrid for Gradient
nx, ny = (10, 10)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
xg, yg = np.meshgrid(x,y)

# calculate the gradient of f(x,y)
# Note: numpy returns answer in rows (y), columns (x) format
Gy, Gx = np.gradient(f(xg, yg))


# Make a Color plot to display the data
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.colorbar()
plt.quiver(xg, yg, Gx, Gy, scale = 1000, color = 'w')
plt.title('Gradient of f(x,y) = xy^2')
plt.show()


#calculate the gradient of f(x,y) = xy^2
def ddx(x,y):
    return y**2

def ddy(x, y):
    return 2*x*y

# np.gradient(f(xg, yg))' nin aynısı
Gx = ddx(xg, yg)
Gy = ddy(xg, yg)

# Make a Color plot to display the data
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.colorbar()
plt.quiver(xg, yg, Gx, Gy, scale = 1000, color = 'w')
plt.title('Plot of [y^2, 2xy]')
plt.show()