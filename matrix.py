import numpy as np

A = np.array([ 
      [1,2,3],
      [3,0,0]])

B = np.array([ 
      [1,0,0,0],
      [0,1,0,0],
      [2,1,1,1]])

M,N = A.shape
N,P = B.shape

C = np.zeros([M,P])

for k in range(M):
    for j in range(P):
        x=0
        for i in range(N):
            x += A[k,i]*B[i,j]
        C[k,j]=x
print(C)