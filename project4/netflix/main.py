import numpy as np
import kmeans
import common
import naive_em
import em
import em2
from typing import NamedTuple, Tuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# X = np.loadtxt("toy_data.txt")

# # TODO: Your code here
# K = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]
# kmeans_mixture = []
# kmeans_post = []
# for k in K:
#     l_cost = float('inf')
#     for seed in seeds:
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         l_cost = min(l_cost, cost)
        
#     # print(f'K={k}, seed={seed}, cost={l_cost}')
#     kmeans_mixture.append(mixture)
#     kmeans_post.append(post)

# K = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]
# em_mixture = []
# em_post = []
# em_log = []
# for k in K:
#     log_l = -1*float('inf')
#     for seed in seeds:
#         mixture, post = common.init(X, k, seed)
#         mixture, post, log = em.run(X, mixture, post)
#         log_l = max(log_l, log)
        
#     # print(f'K={k}, seed={seed}, cost={log_l}')
#     em_mixture.append(mixture)
#     em_post.append(post)
#     em_log.append(log_l)

# for ele in range(1, 5):
#     print(common.bic(X, em_mixture[ele-1], em_log[ele-1]))

# for i in range(1, 5):
#     common.plot(X, kmeans_mixture[i-1], kmeans_post[i-1], f'K={i}')
    
#     common.plot(X, em_mixture[i-1], em_post[i-1], f'K={i}')
    

X = np.loadtxt("netflix_incomplete.txt")
K = [1, 12]
seeds = [0, 1, 2, 3, 4]
em_mixture = []
em_post = []
em_log = []
for k in K:
    log_l = -1*float('inf')
    for seed in seeds:
        mixture, post = common.init(X, k, seed)
        mixture, post, log = em2.run(X, mixture, post)
        log_l = max(log_l, log)
        
    print(f'K={k}, seed={seed}, cost={log_l}')
    em_mixture.append(mixture)
    em_post.append(post)
    em_log.append(log_l)