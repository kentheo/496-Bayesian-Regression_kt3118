import numpy as np
from answers import *
from polynomial import *

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

def bayesian_regression(K, X_train, Y_train, alpha, beta):
    phi_train = polynomial_design_matrix(K, X_train)
    lml1 = lml(alpha, beta, phi_train, Y_train)

    return lml1

print(bayesian_regression(3, X_train, Y_train, 2.0, 4.0))

# alpha = 1.0
# beta = 2.0
#
# N = 100
# M = 50
# Phi = np.ones((N, M))
# Y = np.ones((N, 1))*0.2
#
# term = np.array([[14.0, 14.0], [14.0, 44.0]])
# print(term)
# inv = np.linalg.inv(term)
#
# dett = np.linalg.det(term)
#
#
# y = np.array([[-0.75426779], [-0.5480492]])
# print(y.shape)
#
# ret = -0.5 * np.log(dett) - 0.5 * np.dot(y.T, np.dot(inv, y)) - np.log(2.0*np.pi)
# print ret
