from tqdm import tqdm
from funs import *
import pandas as pd


alpha_grid = np.append(np.arange(0.001, 1, 0.02), 0.999)
# 1st case
w = 3
k = 50
Theta, ThetaB = generate_thetas(w, 100)
results_small = pd.DataFrame({'alpha': alpha_grid,
                        'dtv': 0,
                        'case': 'small'})


for ix, alpha in enumerate(tqdm(results_small['alpha'])):
    X = generate_data(Theta, ThetaB, alpha, k)
    res = run_em(data=X, alpha=alpha)
    theta, thetaB = res['Theta'], res['ThetaB']
    dtv = (calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1 / (w + 1)
    results_small.loc[ix, 'dtv'] = dtv


# 2nd case
w = 6
k = 1000
Theta, ThetaB = generate_thetas(w, 2022)
results_big = pd.DataFrame({'alpha': alpha_grid,
                        'dtv': 0,
                        'case': 'big'})


for ix, alpha in enumerate(tqdm(results_small['alpha'])):
    X = generate_data(Theta, ThetaB, alpha, k)
    res = run_em(data=X, alpha=alpha)
    theta, thetaB = res['Theta'], res['ThetaB']
    dtv = (calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1 / (w + 1)
    results_big.loc[ix, 'dtv'] = dtv

results = pd.concat([results_small, results_big])
results.to_csv('Results/results_diff_alpha.csv', index=False)
