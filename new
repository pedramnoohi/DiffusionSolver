import math

import numpy as np
uHist = np.empty((4 + 1, 2))

D = (np.array([1,2]))
uHist[0,:]=D
s=np.linspace(0,1,5)

def f(x):return math.sin(2*(math.pi)*x)

f=np.vectorize(f)


A = uHist
print(A)
B = np.array([1,2])

print(np.matmul(A, B))

"""
for j in [i for i in range(0, 5 )]:
    D[j, j] = -2
    if j == 0:
        D[j, j + 1] = 1
    elif j == 5-1:
        D[j, j - 1] = 1
    else:
        D[j, j - 1] = 1
        D[j, j + 1] = 1
D[0, 5-1] = 1
D[5-1,0] = 1
print(D)
h=1/(5 +1)
print(1/(h**2))
D=D*(1/(h**2))
print(D)
"""