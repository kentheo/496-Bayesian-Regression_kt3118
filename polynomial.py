import numpy as np
import matplotlib.pyplot as plt

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

# Polynomial Design matrix of size N x (K+1)
def polynomial_design_matrix(K, x):
    phi = np.zeros((len(x), K+1))
    for j in range(K+1):
        for i in range(len(x)):
            phi[i][j] = x[i]**j
    # print("phi shape = {0}".format(phi.shape))
    return phi

# Estimate weights MLE
def weights_MLE(K, X, Y):
    phi_k = polynomial_design_matrix(K, X)
    try:
        inverse = np.linalg.inv(np.dot(phi_k.T, phi_k))
    except np.linalg.LinAlgError:
        print("ERROR!! Matrix not invertible")
        pass
    else:
        temp = np.dot(inverse, phi_k.T)
        w_MLE = np.dot(temp, Y)
        return w_MLE
    # SHOULD NEVER GET HERE
    print("ERROR!!!!! Can't return w_MLE")
    return phi_k

# Estimate sigma MLE
def sigma_MLE(theta, phi, y, x):
    sum = 0
    print("pfffff shape = {0}".format(phi.shape))
    for i in range(len(x)):
        sum += (y[i] - np.dot(theta.T, phi[i]))**2
    print("sum = {0}".format(sum))
    return sum / len(x)

def gaussian_noise(mu, sigma, N):
    noise = np.zeros((N,1))
    # print(mu.shape)
    for i in range(N):
        noise[i] = np.random.normal(mu[i], sigma)
    # print(noise.shape)
    return noise

def linear_regression_polynomial(K, X_train, Y_train, X_test):
    phi_train = polynomial_design_matrix(K, X_train)
    w = weights_MLE(K, X_train, Y_train)
    print("w shape = {0}".format(w.shape))
    sigma = sigma_MLE(w, phi_train, Y_train, X_train)
    print("sigma shape = {0}".format(sigma.shape))

    print("----------------- TEST DATA {0} -------------------".format(K))
    phi_test = polynomial_design_matrix(K, X_test)

    # This bit here is not necessary. DO NOT USE NOISE when finding the new Y
    mu = np.dot(phi_test, w)
    parametered_x = np.dot(phi_test, w)
    noise = gaussian_noise(mu, sigma, len(X_test))

    print("parametered_x shape = {0}".format(parametered_x.shape))
    print("noise shape = {0}".format(noise.shape))
    Y_test = parametered_x
    print("------------------------------------------------")
    return Y_test

def main():
    fig = plt.figure(figsize=(11,9))

    N_test = 200
    X_test = np.reshape(np.linspace(-0.3, 1.3, N_test), (N_test, 1))

    # Linear regrssion for each polynomial value
    K_0 = linear_regression_polynomial(0, X_train, Y_train, X_test)
    K_1 = linear_regression_polynomial(1, X_train, Y_train, X_test)
    K_2 = linear_regression_polynomial(2, X_train, Y_train, X_test)
    K_3 = linear_regression_polynomial(3, X_train, Y_train, X_test)
    K_11 = linear_regression_polynomial(11, X_train, Y_train, X_test)

    plt.plot(X_train, Y_train, 'ko', X_test, K_0, 'b-', X_test, K_1, 'g-', X_test, K_2, 'r-', X_test, K_3, 'm-', X_test, K_11, 'y-')
    plt.legend(["Original", "K = 0", "K = 1", "K = 2", "K = 3", "K = 11"], loc='lower left')
    plt.xlabel("X")
    plt.ylabel("Y")
    axes = plt.gca()
    axes.set_xlim([-0.3,1.3])
    axes.set_ylim([-1.4,3.6])
    plt.show()

    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
