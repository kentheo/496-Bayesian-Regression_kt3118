import numpy as np
from answers import *
from trigonometric_grad_descent import trigonometric_design_matrix
import matplotlib.pyplot as plt

def readFile():
	# Read all max values for alpha and beta from files
	read_lines = []
	# textfilename = '/home/kendeas93/Desktop/496-bayesian-regression_kt3118/trig_lml' + str(K) + '.txt'
	for j in range(11):
		textfilename = '/home/kendeas93/Desktop/496-bayesian-regression_kt3118/trig_lml' + str(j) + '.txt'
		with open(textfilename, 'r') as f:
		    f1 = f.readlines()
		    for i in range(len(f1)):
		    	if i == 1:
		    		x = f1[i].replace("alpha = ", "")
		    		x = x.replace(" beta = ", "")
		    		x = x.replace("\n", "")
		    		g = x.split(",")
		    		f = []
		    		for k in range(2):
		    			f.append(float(g[k]))
		    		read_lines.append(f)

	return read_lines

def getMaxLML(max_params):
	# Data
	N = 25
	X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
	Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

	maxLML_list = []
	for i in range(11):
		alpha = max_params[i][0]
		beta = max_params[i][1]
		phi_train = trigonometric_design_matrix(i, X_train)
		maxLML_list.append(lml(alpha, beta, phi_train, Y_train))

	return maxLML_list

def main():
	max_params = readFile()
	lml_max_values = getMaxLML(max_params)

	# PLOTS
	fig = plt.figure(figsize=(11,9))

	order_of_basis = []
	for i in range(11):
	    order_of_basis.append(i)

	plt.plot(order_of_basis, lml_max_values, 'r-')
	plt.xlabel("Order of basis")
	plt.ylabel("LML")
	plt.title("Max LML against Order of Basis Functions")

	fig.savefig('lml_against_orders.png')

	plt.show()
	pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()