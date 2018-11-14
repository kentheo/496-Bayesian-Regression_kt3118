import numpy as np
import matplotlib.pyplot as plt

def gaussian_design_matrix(K, x, l):
    phi = np.zeros((len(x), K+1))
    # Get means of m_i
    means = np.reshape(np.linspace(-0.5, 1.0, K+1), (K+1, 1))
    for i in range(len(x)):
        phi[i][0] = 1.0
        for j in range(1, K+1):
            term = (x[i] - means[j])**2
            term /= (2*l)**2
            phi[i][j] = np.exp(-term)

    return phi

# Training data
N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

# Get 5 samples from posterior distribution over the weights
alpha = 1.
beta = 0.1
l = 0.1
K = 10
phi_train = gaussian_design_matrix(K, X_train, l)

aI = alpha*np.identity(K+1)

sigma_temp = (beta**-1) * np.dot(phi_train.T, phi_train) + np.linalg.inv(aI)
sigma = np.linalg.inv(sigma_temp)

mu_temp = (beta**-1) * np.dot(phi_train.T, Y_train)
mu = np.dot(sigma, mu_temp)

# print("sigma = {0}, mu = {1}".format(sigma.shape, mu.shape))

# Test data
N_test = 200
X_test = np.reshape(np.linspace(-0.3, 1.3, N_test), (N_test, 1))
phi_test = gaussian_design_matrix(K, X_test, l)

random_w_list, Y_test_list = [], []
for i in range(5):
    random_w = np.random.multivariate_normal(mu.reshape((K+1,)), sigma).reshape(K+1, 1)
    random_w_list.append(random_w)
    Y_test_list.append(np.dot(phi_test, random_w))

# Predictive mean
mu_predictive = np.dot(mu.T, phi_test.T)
sigma_predictive = np.dot(np.dot(phi_test, sigma), phi_test.T)
predictive_mean = np.random.multivariate_normal(mu_predictive.reshape((N_test,)), sigma_predictive).reshape(N_test, 1)

error = np.random.multivariate_normal(mu_predictive.reshape((N_test,)), sigma_predictive).reshape(N_test, 1)
print(error.shape)

# Plots
fig = plt.figure(1,figsize=(11,9))

plt.plot(X_train, Y_train, 'ko')
plt.plot(X_test, Y_test_list[0], 'b-')
plt.plot(X_test, Y_test_list[1], 'g-')
plt.plot(X_test, Y_test_list[2], 'r-')
plt.plot(X_test, Y_test_list[3], 'm-')
plt.plot(X_test, Y_test_list[4], 'c-')
plt.plot(X_test, predictive_mean, 'y-')
plt.fill_between(X_test, predictive_mean-error, predictive_mean+error)
plt.legend(["Original", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5", "Predictive Mean"], loc='lower left')
plt.xlabel("Phi")
plt.ylabel("Y")

plt.show()
