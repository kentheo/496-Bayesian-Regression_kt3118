import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 25
    X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()


X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(X,Y)

x_i = np.array([X[0,0], Y[0,0]])

values2 = np.zeros((len(X), len(Y)))

for i in range(len(X[:,0])):
    for j in range(len(Y[0,:])):
        x_i = np.array([X[i,j], Y[i,j]])
        values2[i,j] = function(x_i)

# Contour Plot of F2
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1, 1, 1)

steps2 = gradient_descent_f2(0.2)
# print(steps2.shape)

CS = ax.contourf(X, Y, values2, 30)

ax.plot(steps2[:,0], steps2[:,1], 'ro-')
plt.title('f2 Contour Plot', fontsize=20)

fig.savefig('f2_contour_step0.2.png')

plt.show()
