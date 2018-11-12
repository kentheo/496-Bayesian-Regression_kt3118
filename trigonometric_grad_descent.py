import numpy as np
from answers import *
import matplotlib.pyplot as plt


# Trigonometric Design matrix of size N x (2K+1)
def trigonometric_design_matrix(K, x):
    phi = np.zeros((len(x), (2*K)+1))
    for i in range(len(x)):
        phi[i][0] = 1.0
        for j in range(1, K+1):
            phi[i][(2*j)-1] = np.sin(2*np.pi*j*x[i])
            phi[i][2*j] = np.cos(2*np.pi*j*x[i])

    return phi

def gradient_descent(step_size, Phi, Y_values):
    start_value = [0.1, 0.15]
    x_i = start_value
    max_iterations = 500
    iteration = 0

    prev_step_size1 = 1
    prev_step_size2 = 1

    steps = np.array([x_i])
    while (iteration < max_iterations):
        prev = x_i
        grad = grad_lml(x_i[0], x_i[1], Phi, Y_values)

        x_i = x_i + step_size * grad.T

        steps = np.concatenate((steps, np.array([x_i])), axis=0)
        iteration += 1

    return [steps, iteration, start_value]

def plot_contour_with_grad_descent(K, step_size):
    # Data
    N = 25
    X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

    phi_train = trigonometric_design_matrix(K, X_train)

    # X = alpha range
    # Y = beta range
    X = np.arange(0.001, 0.5, 0.05)
    Y = np.arange(0.001, 0.5, 0.05)

    X, Y = np.meshgrid(X, Y)

    x_i = np.array([X[1,0], Y[1,0]])

    values = np.zeros((len(X), len(Y)))

    for i in range(len(X[:,0])):
        for j in range(len(Y[0,:])):
            x_i = np.array([X[i,j], Y[i,j]])
            values[i,j] = lml(x_i[0], x_i[1], phi_train, Y_train)


    [steps, iteration, x_i] = gradient_descent(step_size, phi_train, Y_train)

    # print(iteration)
    print(steps[0])
    print("alpha = {0}, beta = {1}".format(steps[len(steps)-1, 0], steps[len(steps)-1, 1]))

    # Contour Plot of F2
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)

    CS = ax.contourf(X, Y, values, 30)

    ax.plot(steps[:,0], steps[:,1], 'ro-')
    plt.title('Log Marginal Likelihood Plot', fontsize=20)
    plt.xlabel('alpha')
    plt.ylabel('beta')

    # Write important stuff to file
    max_alpha = steps[len(steps)-1, 0]
    max_beta = steps[len(steps)-1, 1]
    maxes = "alpha = " + str(max_alpha) + ", beta = " + str(max_beta)

    result_to_write = [x_i, maxes]
    textfilename = '/homes/kt3118/Desktop/496-bayesian-regression_kt3118/trig_lml' + str(K) + '.txt'
    with open(textfilename, 'w+') as f:
        for i in range(len(result_to_write)):
            f.write("%s\n" % str(result_to_write[i]))

    # Save figure
    filename = 'trig_lml' + str(K) + '.png'
    fig.savefig(filename)

    plt.show()

# K, step size
def main():
    plot_contour_with_grad_descent(6, 0.0001)
    pass
# K_list = []
# sigma_list = []
# order_of_basis = []
# for i in range(11):
#     results = linear_regression_trigonometric_CV(i, X_train, Y_train)
#     K_list.append(results[0])
#     sigma_list.append(results[1])
#     order_of_basis.append(i)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
