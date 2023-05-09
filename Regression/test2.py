import pandas as pd
from sklearn import linear_model
model = linear_model.LinearRegression()

wine = pd.read_csv("winequality-red.csv")

X = wine.drop(['quality'], axis=1)
Y = wine['quality']

#線形モデルを訓練事例で学習させ，テスト事例で精度を測る
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)

#標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaler = scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaler = scaler.transform(X_test)

# 予測モデルを作成
model.fit(X_train_scaler, y_train)

#最小二乗誤差でモデルの適合度を評価
from sklearn.metrics import mean_squared_error
y_train_predict = model.predict(X_train_scaler)
y_test_predict = model.predict(X_test_scaler)
print('MSE(train data): ', round(mean_squared_error(y_train, y_train_predict),8))

print('MSE(test data): ', round(mean_squared_error(y_test, y_test_predict),8))
