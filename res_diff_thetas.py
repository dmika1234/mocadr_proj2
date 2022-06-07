import numpy as np
from tqdm import tqdm
from funs import *
import pandas as pd


max_seeds = 200

results_small = pd.DataFrame({'dtv': np.zeros(max_seeds), 'case': 'small'})
w = 6
k = 1000
alpha = 0.5


for ix, seed in enumerate(tqdm(np.arange(max_seeds))):
    Theta, ThetaB = generate_thetas(w, seed=seed)
    X = generate_data(Theta, ThetaB, alpha, k)
    res = run_em(data=X, alpha=alpha)
    theta, thetaB = res['Theta'], res['ThetaB']
    dtv = (calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1 / (w + 1)
    results_small.loc[ix, 'dtv'] = dtv


# results_big = pd.DataFrame({'dtv': np.zeros(max_seeds), 'case': 'big'})
#
# results = pd.concat([results_small, results_big])
results_small.to_csv('Results/results_diff_thetas4.csv', index=False)
