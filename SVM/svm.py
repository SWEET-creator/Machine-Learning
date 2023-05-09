import pandas as pd
from sklearn.preprocessing import StandardScaler

def init():
    data = pd.read_csv('./winequality-red.csv')
    X = data.drop(['quality'], axis=1)
    Y = data['quality']

    # 標準化
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaler = scaler.transform(X)