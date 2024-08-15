import numpy as np
import em2
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = 12
n, d = X.shape
seed = [0, 1, 2, 3, 4]
for s in seed:
    mixture, post = common.init(X, K, s)
    mixture, post, log = em2.run(X, mixture, post)
    X_pred = em2.fill_matrix(X, mixture)
    print(common.rmse(X_gold, X_pred))
# TODO: Your code here
