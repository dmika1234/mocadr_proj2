import numpy as np
from funs import *
import pandas as pd
import random as rng


# max_iter = 700
# w = 3
# k = 50
# alpha = 0.5
# Theta = np.array([[0.375, 0.1, 0.14285714285714285],
#                   [0.125, 0.2, 0.2857142857142857],
#                   [0.25, 0.3, 0.14285714285714285],
#                   [0.25, 0.4, 0.42857142857142855]], )
# ThetaB = np.array([0.25, 0.25, 0.25, 0.25])
#
# X = generate_data(Theta, ThetaB, alpha, k, seed=2022)
#
#
#
# norm, normB = run_em_stop_cond(X, max_iter=max_iter, alpha=alpha)
# results = pd.DataFrame({'iteration': np.arange(max_iter)+1, 'norm': norm, 'normB': normB})
#
# results.to_csv('Results/results_stop_cond_small.csv', index=False)


w = 6
k = 50
alpha = 0.5

rng.seed(1)
Theta = np.zeros((4,w))
Theta[:3, :] = np.random.random((3,w)) / w
Theta[3, :] = 1 - np.sum(Theta, axis=0)
ThetaB = np.zeros(4)
ThetaB[:3] = np.random.random(3) / w
ThetaB[3] = 1 - np.sum(ThetaB, axis=0)

X = generate_data(Theta, ThetaB, alpha, k, seed=2022)
max_iter = 700


norm, normB = run_em_stop_cond(X, max_iter=max_iter, alpha=alpha)
results = pd.DataFrame({'iteration': np.arange(max_iter)+1, 'norm': norm, 'normB': normB})

results.to_csv('Results/results_stop_cond_big_smallk.csv', index=False)
