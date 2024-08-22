import numpy as np

# State = [0, 1, 2, 3, 4]
# Action = [0, 1, 2], representing moving left, staying, moving right respectively

# Transition probability matrix
T = np.array([
    [[1/2, 1/2, 0, 0, 0], [1/2, 1/2, 0, 0, 0], [2/3, 1/3, 0, 0, 0]],
    [[1/3, 2/3, 0, 0, 0], [1/4, 1/2, 1/4, 0, 0], [0, 2/3, 1/3, 0, 0]],
    [[0, 1/3, 2/3, 0, 0], [0, 1/4, 1/2, 1/4, 0], [0, 0, 2/3, 1/3, 0]],
    [[0, 0, 1/3, 2/3, 0], [0, 0, 1/4, 1/2, 1/4], [0, 0, 0, 2/3, 1/3]],
    [[0, 0, 0, 1/3, 2/3], [0, 0, 0, 1/2, 1/2], [0, 0, 0, 1/2, 1/2]],
])

num_state = 5
num_action = 3
gamma = 1/2  # Discount factor

# Initialization
V = np.zeros(num_state)

# Reward vector
R = np.zeros(num_state)
R[4] = 1

num_iter = 10  # Increase number of iterations to ensure convergence

for i in range(num_iter):
    Q = np.zeros((num_state, num_action))
    for s in range(num_state):
        for a in range(num_action):
            for t in range(num_state):
                Q[s][a] += T[s][a][t] * (R[s] + gamma * V[t])
    V = np.max(Q, axis=1)
    print(f"Final Value Function after {num_iter} iterations:")
    print(V)
