import numpy as np

with open('./sample.csv' , 'w') as f:
    N = 10
    for i in range(N):
        for j in range(N):
            f.write(str(int(np.random.uniform(1,10))))
            if(j != N-1):
                f.write(",")
        f.write('\n')

    for i in range(N):
        for j in range(N):
            f.write(str(chr(ord("a")+i%26))+ str(chr(ord("a")+j%26)))
            if(j != N-1):
                f.write(",")
        f.write('\n')

f.close()