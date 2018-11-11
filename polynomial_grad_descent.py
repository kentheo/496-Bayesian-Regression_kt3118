import numpy as np
from answers import *
from polynomial import *


def gradient_descent(step_size, Phi, Y_values, tolerance):
	x_i = [0.25, 0.25]
	max_iterations = 100
	iteration = 0

	prev_step_size1 = 1
	prev_step_size2 = 1

	steps = np.array([x_i])
	while (iteration < max_iterations):
		prev = x_i
		grad = grad_lml(x_i[0], x_i[1], Phi, Y_values)

		x_i = x_i + step_size * grad.T

		prev_step_size1 = abs(x_i[0] - prev[0])
		prev_step_size2 = abs(x_i[1] - prev[1])
		steps = np.concatenate((steps, np.array([x_i])), axis=0)
		iteration += 1
	    
	return [steps, iteration]

def bayesian_regression(K, X_train, Y_train, alpha, beta):
    phi_train = polynomial_design_matrix(K, X_train)
    lml1 = lml(alpha, beta, phi_train, Y_train)

    grad = grad_lml(alpha, beta, phi_train, Y_train)

    return [lml1, grad]

def plot_contour_with_grad_descent(K, step_size, tolerance):
	# Data
	N = 25
	X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
	Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

	phi_train = polynomial_design_matrix(K, X_train)

	# X = alpha range
	# Y = beta range
	X = np.arange(0.1, 2.0, 0.1)
	Y = np.arange(0.1, 2.0, 0.1)

	X, Y = np.meshgrid(X, Y)

	x_i = np.array([X[1,0], Y[1,0]])

	values = np.zeros((len(X), len(Y)))

	for i in range(len(X[:,0])):
	    for j in range(len(Y[0,:])):
	        x_i = np.array([X[i,j], Y[i,j]])
	        values[i,j] = lml(x_i[0], x_i[1], phi_train, Y_train)


	[steps, iteration] = gradient_descent(step_size, phi_train, Y_train, tolerance)

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

	fig.savefig('lml.png')

	plt.show()

# K, step size, tolerance
plot_contour_with_grad_descent(1, 0.01, 0.001)
# print(bayesian_regression(1, X_train, Y_train, 2.0, 4.0))

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
