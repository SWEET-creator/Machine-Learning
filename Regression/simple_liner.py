from cmath import inf
from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def gauss_plot(ax):
    x = np.arange(-5, 5, 0.1)
    mu=0
    sigma=1
    a =1
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    label_txt = "μ=" + str(mu) + ", σ=" + str(sigma)
    ax.plot(x, gauss(x, a, mu, sigma), label=label_txt)

def numerical_diff(f,x):
    h = 1.0E-4
    return (f(x+h)-f(x-h))/(2*h)

def MSE(y, y_prev):
    return sum((y - y_prev)**2)

def liner_model(a, b):
    return a*x + b

def simple_liner(x, y):
    mse = np.inf
    a = 1
    b = 1
    while(0.01 < mse):
        y_prev = liner_model(a, b)
        mse = MSE(y ,y_prev)
        a = a - numerical_diff(liner_model(a, b),a)
        b = b - numerical_diff(MSE,b)
    return a,b
    
fig = plt.figure()
ax = fig.add_subplot()
ax.grid()

N = 50

x = np.random.normal(0, 1,N)
eps = np.random.normal(0, 0.5,N)

y = x + eps
ax.plot(x,y,"o", label="data")

a,b = simple_liner(x, y)

y_prev = a*x + b
ax.plot(x,y_prev, label="data")

ax.legend()
plt.show()