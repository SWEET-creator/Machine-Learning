import pandas as pd
import numpy as np
 
wine = pd.read_csv("icecream.csv")

# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
model = linear_model.LinearRegression()

X = wine.drop(['year', 'month', 'spending_amount'], axis=1)

# 標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaler = scaler.transform(X)

Y = wine['spending_amount']

# 予測モデルを作成
model.fit(X_scaler, Y)

# 回帰係数
print(model.coef_)
 
# 切片 (誤差)
print(model.intercept_)
 
# 決定係数
print(model.score(X_scaler, Y))