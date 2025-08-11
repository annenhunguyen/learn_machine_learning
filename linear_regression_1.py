import numpy as np

def solve_parameters(X,Y):
    X_transpose = X.T
    X_Xtransposed = X_transpose@X
    X_XT_inversed = np.linalg.inv(X_Xtransposed)
    W = Y.T @ X @ X_XT_inversed
    return W

#2x+3
X = np.array([
    [1,831,4,1],
    [2,1276,2,1],
    [2,1159,1,1],
    [1,742,3,1],
    [2,1050,3,1]
])

Y = np.array([2895,3950,3640,2750,3450])

W = solve_parameters(X,Y)

def predict_result(A):
    # A = np.array(A)
    prediction = A@W
    print("predicted result is",prediction)

A = np.array([2,1050,3,1])
predict_result(A)

print(A.shape)
print(W.shape)
