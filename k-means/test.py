import numpy as np

x = np.array([3, -2])
xx = [1, 2]
xxx = [5, -4]
xxxx = [2, 2]

c1 = np.array([2,-2])
c2 = np.array([3,2])

print(np.linalg.norm(x-c1))
print(np.linalg.norm(x-c2))