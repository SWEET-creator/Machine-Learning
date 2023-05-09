from typing import Sized
import numpy as np
import collections
import matplotlib.pyplot as plt

def distance(x, y):
    return np.linalg.norm(x-y,ord=2)

def k_NN(train_data, test_data, k):
    pred_data = []
    for td in test_data:
        dis_list = []
        index = 0
        N = len(train_data[0])
        for j in train_data:
            dis_list.append([distance(j[0:N-1], td[0:N-1]),train_data[index][N-1]])
            index += 1
        dis_list.sort(key=lambda x: x[0])
        l = []
        for i in range(k):
            l.append(dis_list[i][1])
        c = collections.Counter(l)
        pred_data.append(c.most_common(1)[0][0])

    return pred_data

def accuracy_score(test_data, pred_data):
    return sum(test_data[i][-1] == pred_data[i] for i in range(len(test_data)))/len(test_data)

def my_plt(test_data, train_data, pred_data):
    Cluster_color = {
            0:"tab:red",
            1:"tab:blue",
    }
            
    for x in train_data:
        plt.plot(x[0], x[1], marker='.',linewidth=False, color=Cluster_color[x[-1]],label="train data")
    for x in test_data:
        plt.plot(x[0], x[1], marker='x',linewidth=False, color=Cluster_color[x[-1]],ms = 10, label = "test data")
    for x, i in zip(test_data,range(len(test_data))):
        plt.plot(x[0], x[1], marker='+',linewidth=False, color=Cluster_color[pred_data[i]],ms = 10, label = "pred")
    
    plt.show()

def main():
    train_data =np.array([
        [1,2,0],
        [2,1,0],
        [1,1,0],
        [3,4,1],
        [4,3,1],
        [3,3,1]
    ])

    test_data = np.array([
        [2,2,0],
        [4,4,1],
        [2,3,1],
        [2.5,2,1],
        [2.5,2.5,0]
    ])

    pred_data = k_NN(train_data, test_data, k = 2)
    print("acuracy:", accuracy_score(test_data, pred_data)*100, "%")
    my_plt(test_data, train_data, pred_data)

main()