import numpy as np

# Define the input data
labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
coordinates = np.array([(0, 0), (2, 0), (1, 1), (0, 2), (3, 3), (4, 1), (5, 2), (1, 4), (4, 4), (5, 5)])
perceptron_mistakes = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])
# Calculate theta_0
theta_0 = np.sum(perceptron_mistakes * labels)

# Calculate the feature map
def phi(x):
    x1, x2 = x
    return np.array([x1**2, np.sqrt(2) * x1 * x2, x2**2])

# Calculate theta
theta = np.zeros(3)
for i in range(len(labels)):
    theta += perceptron_mistakes[i] * labels[i] * phi(coordinates[i])

# Print the results
print(f"theta_0 = {theta_0:.2f}")
print(f"theta = [{theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.2f}]")