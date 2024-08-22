import numpy as np
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1.0)
X = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
clf.fit(X, y)

    # Extract theta and theta_0
theta = clf.coef_[0]
theta_0 = clf.intercept_[0] 
print(f"Weight vector (theta): {theta}")
print(f"Intercept term (theta_0): {theta_0}")
margin = 2 / np.linalg.norm(theta)
print(f"Margin: {margin}")
decision_function = X.dot(theta/2) + theta_0/2
hinge_losses = np.maximum(0, 1 - y * decision_function)
total_hinge_loss = np.sum(hinge_losses)
print(f"Total hinge loss: {total_hinge_loss}")