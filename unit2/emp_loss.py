import numpy as np
theta = np.array([0,1,2])
x = np.array([[1,0,1], [1,1,1], [1,1,-1], [-1,1,1]])
y = np.array([2, 2.7, -0.7, 2])

def loss(arg):
    return min(0, arg)
h_loss = 0
h_loss += loss( 1 - (y - np.dot(x, theta)) )

print(h_loss/4)