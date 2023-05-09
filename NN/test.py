import numpy as np

A = np.array([[1,2]])
B = np.array([[5,6],[7,8]])

print(np.dot(A,B))

for i in range(1):
    S = 0
    for j in range(2):
        S += A[j] * B[j][i]
    print(S)