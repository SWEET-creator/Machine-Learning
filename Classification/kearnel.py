import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

def linear(x1, x2):
    return np.dot(x1,x2)

def RBF(x1, x2):
    sig = 1
    return np.exp(-abs(x1 - x2)**2/2*sig)

x = [np.arange(0,10,.1),np.arange(0,10,.1)]
y = [np.arange(0,10,.1),np.arange(0,10,.1)]
axis = np.arange(0,10,0.1)
for i in range(len(axis)):
    for j in range(len(axis)):
        ax.plot(axis, axis, RBF(axis,axis), label='parametric curve')

plt.show()