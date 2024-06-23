
import numpy as np
T = 1000

def perceptron(X, Y):
    mistakes = 0
    theta_values = []
    n = Y.shape[0]
    theta = np.zeros(X.shape[1])
    theta_not = 0
    
    while mistakes <=4:
        for i in range(n):
            if Y[i] * np.dot(theta, X[i]) <= 0:
                theta += Y[i] * X[i]
                mistakes += 1
                theta_values.append(theta.copy())
                theta_not += Y[i]
                if mistakes >= 4:
                    return mistakes, theta, theta_not
    
    return mistakes, theta, theta_not
def perceptron2(X, Y):
    mistakes = 0
    n = Y.shape[0]
    theta = np.array([-3,3])
    theta_not = -3
    
    for _ in range(T):
        for i in range(n):
            if Y[i] * (np.dot(theta, X[i])+theta_not) <= 0:
                theta += Y[i] * X[i]
                mistakes += 1
                theta_not += Y[i]
    
    return mistakes, theta, theta_not

X = np.array([[-4, 2], [-2, 1], [-1, -1],[2, 2],[1, -2]])
Y = np.array([1, 1, -1, -1, -1])
X1 = np.array([[-1, 1],[1, -1],[1, 1], [2,2]])
Y1 = np.array([1, 1, -1, -1])
print(perceptron2(X1, Y1))
