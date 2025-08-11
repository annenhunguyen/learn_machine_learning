import numpy as np


def solve_parameters(X,Y,Wi):
    dL_dW_atWi = 2*X.T @ (X@Wi -Y)
    alpha = 1e-2
    sum_dLdW_atWi = dL_dW_atWi.mean()
    print(sum_dLdW_atWi)
    i=1
    while i<50000:   #sum_dLdW_atWi != 0
        W_i1 = Wi - alpha * dL_dW_atWi
        dL_dW_atWi1 = 2*X.T @ (X @W_i1 -Y)
        sum_dldW_atWi1 = dL_dW_atWi1.mean()
        if i %5000 == 0:
            print(f"======iteration {i}\n, Gradient-mean: {sum_dldW_atWi1} -- Alpha: { alpha} -- Weight: {Wi}" )
        if abs(sum_dldW_atWi1) > abs(sum_dLdW_atWi):
            alpha = alpha *1e-2
        else:
            dL_dW_atWi = dL_dW_atWi1
            Wi = W_i1
            i += 1
    else:
        return Wi

def gradient_descent(X, Y, W):
    
    num_iterations = 50000
    alpha = 1e-02
    for i in range(num_iterations):
        loss = ((X@W-Y)**2).mean()
        gradient = 2*X.T @ (X @W -Y) / len(Y)
        W = W - alpha * gradient
        if i%500==0:
            print(f"Iteration {i} -- loss: {loss} -- gradient:{gradient.mean()}")
    
    return W


X = np.array([
    [1,2,1],
    [3,-1,1],
    [0.5,7,1]
])

Y = np.array([
    [35,22,76.5]
]).reshape(3,1)

W0 = np.random.rand(3, 1)

W = solve_parameters(X,Y,W0)
print(W)

print("end code")