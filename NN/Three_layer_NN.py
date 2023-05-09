import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class sigmoid:
    def forward(x):
        sigmoid_range = 34.538776394910684
        return 1.0 / (1.0 + np.exp(-np.clip(x, -sigmoid_range, sigmoid_range)))

    def backward(y):
        return y * (1 - y)
    

class Three_layer_NN:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.lr = lr
        self.w_1 = np.random.normal(0.0, 1.0, (hidden_size, input_size))
        self.w_2 = np.random.normal(0.0, 1.0, (output_size, hidden_size))

    def feedforward(self, i_data):
        #入力
        x_1 = np.array(i_data, ndmin=2).T

        #隠れ層
        x_2 = np.dot(self.w_1, x_1)
        out_h = sigmoid.forward(x_2)

        #出力層
        x_3 = np.dot(self.w_2, out_h)
        out_o = sigmoid.forward(x_3)

        return out_o

    def backpropagation(self, i_data, t_data):
        x_1 = np.array(i_data, ndmin=2).T
        t = np.array(t_data, ndmin=2).T

        # 隠れ層
        x_2 = np.dot(self.w_1, x_1)
        out_h = sigmoid.forward(x_2)

        # 出力層
        x_3 = np.dot(self.w_2, out_h)
        out_o = sigmoid.forward(x_3)

        # 誤差計算
        error_out = (t - out_o)
        error_hidden = np.dot(self.w_2.T, error_out)

        # 重みの更新
        self.w_2 += self.lr * np.dot((error_out * sigmoid.backward(out_o)), out_h.T)
        self.w_1 += self.lr * np.dot((error_hidden * sigmoid.backward(out_h)), x_1.T)

    

#パラメータ
input_size = 300
hidden_size= 100
output_size = 300
lr = 0.3

# ニューラルネットワークの初期化
nn = Three_layer_NN(input_size, hidden_size, output_size, lr)

n = 300
x0 = np.linspace(-2.0, 2.0, n)
y0 = 0.1*(x0**3 + x0**2 + 0.1*x0 + 1)

training_data_list = x0
test_data_list = y0

# 学習
epoch = 2
for e in range(epoch):
    print('epoch ', e)
    data_size = len(training_data_list)
    for i in range(data_size):
        nn.backpropagation(training_data_list, test_data_list)

y_predict = nn.feedforward(training_data_list)

plt.plot(x0, y0, label="original function")
plt.plot(x0, y_predict, label="predict function")
plt.legend()
plt.savefig("graph.png")

#making animation
fig = plt.figure()
ims = []
plt.plot(x0, y0, label="original function")

for e in range(epoch):
    print('epoch ', e)
    data_size = len(training_data_list)
    for i in range(data_size):
        nn.backpropagation(training_data_list, test_data_list)
    ims.append(plt.plot(x0, nn.feedforward(training_data_list), color="orange"))


anim = animation.ArtistAnimation(fig, ims, interval = 800)
plt.legend()
anim.save('anim.gif')