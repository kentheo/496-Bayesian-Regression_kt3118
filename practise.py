import numpy as np
from answers import *

alpha = 1.0
beta = 2.0

N = 100
M = 50
Phi = np.ones((N, M))
Y = np.ones((N, 1))*0.2

print lml(alpha, beta, Phi, Y)
