from cmath import inf
from ctypes import sizeof
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def PCA():
    ims = []
    return ims

#初期設定
fig = plt.figure(facecolor="k")
ax = plt.subplot()
ax.set_facecolor("k")
ax.spines['bottom'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['right'].set_color('w')
ax.spines['left'].set_color('w')
ax.tick_params(axis='x', colors='w')
ax.tick_params(axis='y', colors='w')
ax.yaxis.label.set_color('w')
ax.xaxis.label.set_color('w')
ax.title.set_color('w')
ax.grid(color='gray',linestyle='--')

#データ作成
N = 50
x = np.linspace(-5,5,N)
data = np.random.normal(0,0.1,N)*x + np.random.normal(0,7,N)
plt.plot(x,data,"oc")

X = np.array([x,data])
cov_X = np.cov(X)

w,v = LA.eig(cov_X)
print(w,v)
plt.arrow(0,0,v[1][0],v[1][1],color="w",head_width=0.3,head_length=0.3)
plt.arrow(0,0,v[0][0],v[0][1],color="w",head_width=0.3,head_length=0.3)

#アニメーションプロット
#ims = PCA()
#ani = animation.ArtistAnimation(fig, ims, interval=500)
plt.show()