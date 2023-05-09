import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def writeFILE(filename):
    with open(filename, 'w') as f:
        for i in range(100):
            for j in range(2):
                x = np.random.uniform(0,10)
                f.write(str(x)+", ")
            f.write("\n")
    f.close()

def readFILE(filename):
    x = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(', ')
            if '\n' in data:
                data.remove('\n')
            x.append([float(x) for x in data])
    f.close()
    return x

def distance(x, y):
    difarrence = [x[i] - y[i] for i in range(len(x))]
    return np.dot(difarrence,difarrence)

def k_means(x, k):
    N = len(x)
    dim = len(x[0])
    z = [0 for i in range(N)]
    z_previous = z[:]
    #　ここはスライスでもできた
    result_data = [x,[],[]]
    
    # Initialize the cluster representative point
    '''
    Mu = [[0 for i in range(dim)]for j in range(k)]
    for i in range(k):
        for j in range(N):
            if x[j] not in Mu:
                Mu[i] = x[j][:]
                break
    '''
    Mu = [[2,-2],[3,2]]
    while(1):
        for i in range(N):
            z_previous[i] = z[i]
            # calculate distance
            z[i] = np.argmin([distance(x[i], Mu[j]) for j in range(k)])
        print("z",z)
        print("Mu", Mu)
        
        #　updated cluster representative points
        Mu_prime = [[[] for j in range(dim)] for i in range(k)]
        for i in range(N):
            for j in range(dim):
                Mu_prime[z[i]][j].append(x[i][j])
        Mu = [[np.average(Mu_prime[i][j]) for j in range(dim)] for i in range(k)]

        result_data[1].append(z[:])
        result_data[2].append(Mu[:])
        if (z == z_previous):
            break
    return result_data

def graphical1D(x,z,Mu):
    fig = plt.figure()
    Cluster_list = [[] for i in range(len(Mu))]
    for i in range(len(x)):
        Cluster_list[z[i]].append(x[i])
    
    for Cluster, number in zip(Cluster_list, range(len(Cluster_list))):
        Cluster_x = [x for x in Cluster]
        Cluster_y = [x for x in Cluster]
        plt.plot(x, [0 for i in range(len(x))], marker='.',linewidth=0,label= "Cluster" + str(number+1))

    plt.plot(Mu,[0 for i in range(len(Mu))],marker="x",linewidth=0,markersize=15, label="Centroids")
    plt.title("Cluster Assignments and Centroids")
    plt.legend()
    fig.savefig('k-means.')

def graphical2D(x,z,Mu):
    fig = plt.figure()
    Cluster_list = [[] for i in range(len(Mu))]
    for i in range(len(x)):
        Cluster_list[z[i]].append(x[i])
    
    for Cluster, number in zip(Cluster_list, range(len(Cluster_list))):
        Cluster_x = [x[0] for x in Cluster]
        Cluster_y = [x[1] for x in Cluster]
        plt.plot(Cluster_x, Cluster_y, marker='.',linewidth=0,label= "Cluster" + str(number+1))

    Mu_x = [i[0] for i in Mu]
    Mu_y = [i[1] for i in Mu]
    plt.plot(Mu_x,Mu_y,marker="x",linewidth=0,markersize=15, label="Centroids")
    plt.title("Cluster Assignments and Centroids")
    plt.legend()
    fig.savefig('k-means.')

def graphical3D(x,z,Mu):
    fig = plt.figure()
    ax = Axes3D(fig)
    Cluster_list = [[] for i in range(len(Mu))]
    for i in range(len(x)):
        Cluster_list[z[i]].append(x[i])
    
    for Cluster, number in zip(Cluster_list, range(len(Cluster_list))):
        Cluster_x = [x[0] for x in Cluster]
        Cluster_y = [x[1] for x in Cluster]
        Cluster_z = [x[2] for x in Cluster]
        plt.plot(Cluster_x, Cluster_y, Cluster_z, marker='.',linewidth=0,label= "Cluster" + str(number+1))

    Mu_x = [i[0] for i in Mu]
    Mu_y = [i[1] for i in Mu]
    Mu_z = [i[2] for i in Mu]
    plt.plot(Mu_x,Mu_y,Mu_z, marker="x",linewidth=0,markersize=15, label="Centroids")
    plt.title("Cluster Assignments and Centroids")
    plt.legend()
    fig.savefig('k-means.')

class get_animation():
    def __init__(self,x,z,Mu):
        self.x = x
        self.z = z
        self.Mu = Mu
        self.animfig = plt.figure()
        artists = []
        if len(x[0]) == 2:
            for step in range(len(z)):
                artists.append(self.get_plt_2D(x,z[step],Mu[step],step))
        elif len(x[0]) == 3:
            self.ax = Axes3D(self.animfig)
            for step in range(len(z)):
                artists.append(self.get_plt_3D(x,z[step],Mu[step],step))
        anim = ArtistAnimation(self.animfig, artists, interval = 1000)
        plt.show()
        anim.save('anim.gif', writer="imagemagick")
    
    def get_plt_2D(self,x,z,Mu,step):
        Cluster_list = [[] for i in range(len(Mu))]
        Cluster_color = {
            0:"tab:red",
            1:"tab:blue",
            2:"tab:green",
            3:'tab:orange',
            4:'tab:purple',
            5:'tab:brown',
            6:'tab:pink',
            7:'tab:gray'}
        # クラスタ数に合わせてカラーの辞書を増やす必要あり

        for i in range(len(x)):
            Cluster_list[z[i]].append(x[i])
        img = [] 
        for Cluster, number in zip(Cluster_list, range(len(Cluster_list))):
            Cluster_x = [x[0] for x in Cluster]
            Cluster_y = [x[1] for x in Cluster]
            img += plt.plot(Cluster_x, Cluster_y, marker='.',linewidth=0,label= "Cluster" + str(number+1), color=Cluster_color[number])
        
        for i, number in zip(Mu, range(len(Mu))):
            Mu_x = i[0]
            Mu_y = i[1]
            img += plt.plot(Mu_x,Mu_y,marker="x",linewidth=10,markersize=15, label="Centroids", color=Cluster_color[number])
        img.append(plt.text(5, 11, "step:"+f'{step+1}',ha='center', fontsize= 15))
        return img
    
    def get_plt_3D(self,x,z,Mu,step):
        Cluster_list = [[] for i in range(len(Mu))]
        Cluster_color = {
            0:"tab:red",
            1:"tab:blue",
            2:"tab:green",
            3:'tab:orange',
            4:'tab:purple',
            5:'tab:brown',
            6:'tab:pink',
            7:'tab:gray'}
        # クラスタ数に合わせてカラーの辞書を増やす必要あり

        for i in range(len(x)):
            Cluster_list[z[i]].append(x[i])
        img = [] 
        for Cluster, number in zip(Cluster_list, range(len(Cluster_list))):
            Cluster_x = [x[0] for x in Cluster]
            Cluster_y = [x[1] for x in Cluster]
            Cluster_z = [x[2] for x in Cluster]
            img += plt.plot(Cluster_x, Cluster_y, Cluster_z, marker='.',linewidth=0,label= "Cluster" + str(number+1), color=Cluster_color[number])
        
        for i, number in zip(Mu, range(len(Mu))):
            Mu_x = i[0]
            Mu_y = i[1]
            Mu_z = i[2]
            img += plt.plot(Mu_x,Mu_y,Mu_z, marker="x",linewidth=0,markersize=15, label="Centroids", color=Cluster_color[number])
            img.append(self.ax.text2D(0.1, 0.95, f'step: {step+1}',ha='center', fontsize= 15, transform=self.ax.transAxes))
        return img

def main(argv):
    # generate data set and write to file
    #writeFILE(argv[1])
    # read data set from file
    x = readFILE(argv[1])
    # Number of cluster
    k = int(argv[2])
    #アニメーションは3まで対応

    # K-means
    result_data = k_means(x, k)

    # Show result
    z = []
    Mu = []
    for i in range(len(result_data[1])):
        z.append(result_data[1][i])
        Mu.append(result_data[2][i])
    if len(x[0]) == 1:
        graphical1D(x,z[-1],Mu[-1])
    elif len(x[0]) == 2:
        graphical2D(x,z[-1],Mu[-1])
        get_animation(x,z,Mu)
    elif len(x[0]) == 3:
        graphical3D(x,z[-1],Mu[-1])
        get_animation(x,z,Mu)

main(sys.argv)