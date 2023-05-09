import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def preprocessing(X, Y):
    #訓練事例とテスト事例に分ける
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    #標準化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test_scaler = scaler.transform(X_test)

    return X_train_scaler, X_test_scaler, y_train, y_test

def MSE(y_train, y_train_predict, y_test, y_test_predict):
    MSE_train = round(mean_squared_error(y_train, y_train_predict),8)
    MSE_train_list.append(MSE_train)
    MSE_test = round(mean_squared_error(y_test, y_test_predict),8)
    MSE_test_list.append(MSE_test)
    print(MSE_train, MSE_test)

def fit_models(X_train_scaler, X_test_scaler, y_train, y_test):
    #線形回帰
    from sklearn import linear_model
    linear = linear_model.LinearRegression()

    # 予測モデルを作成
    linear.fit(X_train_scaler, y_train)

    #最小二乗誤差でモデルの適合度を評価
    y_train_predict = linear.predict(X_train_scaler)
    y_test_predict = linear.predict(X_test_scaler)
    MSE(y_train, y_train_predict , y_test, y_test_predict)

    #リッジ回帰
    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(X_train_scaler, y_train)

    #最小二乗誤差でモデルの適合度を評価
    y_train_predict = ridge.predict(X_train_scaler)
    y_test_predict = ridge.predict(X_test_scaler)
    MSE(y_train, y_train_predict, y_test, y_test_predict)

    #回帰SVM　線形カーネル
    from sklearn.svm import SVR
    SVR_linear = SVR(C=1.0, kernel='linear', epsilon=0.1)

    # 予測モデルを作成
    SVR_linear.fit(X_train_scaler, y_train)

    #最小二乗誤差でモデルの適合度を評価
    y_train_predict = SVR_linear.predict(X_train_scaler)
    y_test_predict = SVR_linear.predict(X_test_scaler)
    MSE(y_train, y_train_predict , y_test, y_test_predict)

    #回帰SVM 多項式カーネル
    SVR_poly = SVR(kernel='poly', C=1, epsilon=0.1)

    # 予測モデルを作成
    SVR_poly.fit(X_train_scaler, np.ravel(y_train))

    #最小二乗誤差でモデルの適合度を評価
    y_train_predict = SVR_poly.predict(X_train_scaler)
    y_test_predict = SVR_poly.predict(X_test_scaler)
    MSE(y_train, y_train_predict , y_test, y_test_predict)

    #回帰SVM RBFカーネル
    SVR_rbf = SVR(kernel='rbf', C=1, epsilon=0.1)

    # 予測モデルを作成
    SVR_rbf.fit(X_train_scaler, np.ravel(y_train))

    #最小二乗誤差でモデルの適合度を評価
    y_train_predict = SVR_rbf.predict(X_train_scaler)
    y_test_predict = SVR_rbf.predict(X_test_scaler)
    MSE(y_train, y_train_predict , y_test, y_test_predict)



print('MSE(train data) MSE(test data)')
MSE_train_list = []
MSE_test_list = []

wine = pd.read_csv("winequality-red.csv")

X = wine.drop(['quality'], axis=1)
Y = wine['quality']

X_train_scaler, X_test_scaler, y_train, y_test = preprocessing(X, Y)

fit_models(X_train_scaler, X_test_scaler, y_train, y_test)


MSE_test_list_rep = []
for i in range(10):
    MSE_train_list = []
    MSE_test_list = []
    X_train_scaler, X_test_scaler, y_train, y_test = preprocessing(X, Y)

    fit_models(X_train_scaler, X_test_scaler, y_train, y_test)

    MSE_test_list_rep.append(MSE_test_list)
    


label = ["線形",
        "リッジ",
        "SVM(kernel = “liner”)",
        "SVM(kernel = “poly”)",
        "SVM(kernel = “rbf”)"]

x = [1,2]
values = ["MSE(train data)", "MSE(test data)"]
plt.ylim(0.3,0.5)
plt.plot(x,[MSE_train_list, MSE_test_list])
plt.xticks(x,values)
plt.legend(label, prop={"family":"Hiragino Mincho ProN"})
plt.savefig("MSE.jpg")

fig = plt.figure()
x = range(10)
plt.ylim(0.3,0.5)
plt.plot(x, MSE_test_list_rep)
plt.legend(label, prop={"family":"Hiragino Mincho ProN"})
plt.savefig("MSE_rep.jpg")
plt.show()
