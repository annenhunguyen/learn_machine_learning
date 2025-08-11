import matplotlib.pyplot as plt
import numpy as np

def polynomial (x):
    y_var = x**3
    return y_var

def log_func(x):
    y_var = np.log(x)
    return y_var

def sigmoid(x):
    y_var = 1/(1+np.exp(-x))
    return y_var

x1 = np.linspace(-2,2,100)
y1 = polynomial(x1)
x2 = np.linspace(0.1,10)
y2 = log_func(x2)
x3 = np.linspace(-5,5)
y3 = sigmoid(x3)

plt.plot(x1, y1)
plt.plot(x2,y2)
plt.plot(x3,y3)

# ax.spines['left'].set_position('zero')    # Move the left spine to x=0
# ax.spines['bottom'].set_position('zero') 

plt.show()