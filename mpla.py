import numpy as np


X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(X,Y)

x_i = np.array([X[1,0], Y[1,0]])

print(x_i[0])