import numpy as np

T = 100

def perceptron(X, Y):
    mistakes = 0
    theta_values = []
    n = Y.shape[0]
    theta = np.zeros(X.shape[1])
    
    for t in range(T):
        for i in range(n):
            if Y[i] * np.dot(theta, X[i]) <= 0:
                theta += Y[i] * X[i]
                mistakes += 1
                theta_values.append(theta.copy())  # Append a copy of the current theta to avoid mutation issues
    
    return mistakes, theta_values

X = np.array([[-1, -1], [1, 0], [-1, 10]])
Y = np.array([1, -1, 1])
X1 = np.array([[1, 0], [-1, 10],[-1, -1]])
Y1 = np.array([ -1, 1,1])
print(perceptron(X1, Y1))
