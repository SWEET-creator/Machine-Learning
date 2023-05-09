import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import copy

num_point = 100  # サンプル値の数
num_c = 2  # クラスタの数

# サンプル値の座標を乱数で初期化
p = np.random.randint(0, 100, (num_point, 3))

c1 = np.random.randint(0, 100, 3)  # クラスタ1の中央の座標
c2 = np.random.randint(0, 100, 3)

count = [0]  # 試行回数をカウントする用の変数
z_new = np.array([0 for x in range(num_point)])  # 各点の状態を保存。０ならc1に近い、１ならc2に近い
z_past = np.array([2 for x in range(num_point)])  # 過去の点の状態を記憶する箱(初期値は2、比較用）
c1_data = []  # 秒数ごとのｃ1のデータ一覧
c2_data = [] # 秒数ごとのｃ2のデータ一覧
z_data = []  # 秒数ごとのｚ_newのデータ一覧
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

def k_means():
    # k-means法を行う
    while np.any(z_new != z_past):
        c1_data.append(copy.deepcopy(c1))  # Cのデータを一覧に格納
        c2_data.append(copy.deepcopy(c2))  # Cのデータを一覧に格納
        for i in range(len(z_new)):  # 最新の状態を過去の状態にコピーする
            z_past[i] = (z_new[i])
        c1_group = []
        c2_group = []
        # それぞれのｃとｘの距離を比べる
        for i in range(num_point):
            for j in range(num_c):
                if j == 0:  # 最初はｘはｃ１に近い点だと仮定する
                    z_new[i] = 0
                    e = abs((p[i][0]-c1[0])**2+(p[i][1]-c1[1])
                            ** 2 + (p[i][2]-c1[2])**2)  # eにxとcの距離を代入
                # もしeがｃ２とｘの距離より大きかったらzの値を更新
                elif (e > abs((p[i][0]-c2[0])**2+(p[i][1]-c2[1])**2+(p[i][2]-c2[2])**2)):
                    z_new[i] = j
        z_data.append(z_new[:])

        # xがc1とc2のどちらに属しているか判定する
        for i in range(num_point):
            if z_new[i] == 0:
                c1_group.append(p[i])
            else:
                c2_group.append(p[i])

        c_c = [0, 0, 0]
        # cの値を属しているサンプル値の平均値に再計算する
        for j in range(len(c1_group)):
            for k in range(3):
                c_c[k] += c1_group[j][k]

        for j in range(len(c_c)):
            c1[j] = c_c[j]/len(c1_group)

        c_c = [0, 0, 0]
        for j in range(len(c2_group)):
            for k in range(3):
                c_c[k] += c2_group[j][k]

        for j in range(len(c_c)):
            c2[j] = c_c[j]/len(c2_group)

        count[0] += 1

    # アニメーションを作る
    ani = FuncAnimation(fig, k_means_plot, frames=count[0], interval=200)

    plt.show()
    ani.save("kmeans.gif", writer="Imagemagick")


def k_means_plot(frame):
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(100, 0)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    for l in range(num_point):
        if z_data[frame][l] == 0:  # c1に属しているなら赤
            ax.plot(p[l][0], p[l][1], p[l][2], marker=".", color="red")
        elif z_data[frame][l] == 1:  # c2に属しているなら青
            ax.plot(p[l][0], p[l][1], p[l][2], marker=".", color="blue")
        ax.plot(c1_data[frame][0],c1_data[frame][1],c1_data[frame][2], marker=".", color="black")  # c1は黒
        ax.plot(c2_data[frame][0],c2_data[frame][1],c2_data[frame][2], marker=".", color="green")  # c2は緑

k_means()
