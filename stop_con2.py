from tqdm import tqdm
from funs import *
import pandas as pd


max_iter = 750
alpha = 0.5
min_diff_grid = np.geomspace(1e-5, 1e-17, num=50)
# 1st case
w = 3
k = 50
Theta, ThetaB = generate_thetas(w, 100)
X = generate_data(Theta, ThetaB, alpha, k)
results_small = pd.DataFrame({'min_diff': min_diff_grid,
                        'dtv': 0,
                        'case': 'small'})

for ix, diff in enumerate(tqdm(results_small['min_diff'])):
    print(ix, diff)
    res = run_em(data=X, alpha=alpha, min_diff=diff, max_iter=max_iter)
    theta, thetaB = res['Theta'], res['ThetaB']
    dtv = (w / 2 * calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1 / (w + 1)
    dtv
    results_small.loc[ix, 'dtv'] = dtv


# 2nd case
w = 6
k = 1000
Theta, ThetaB = generate_thetas(w, 2022)
X = generate_data(Theta, ThetaB, alpha, k)
results_big = pd.DataFrame({'min_diff': min_diff_grid,
                        'dtv': 0,
                        'case': 'big'})


for ix, diff in enumerate(tqdm(results_big['min_diff'])):
    res = run_em(data=X, alpha=alpha, min_diff=diff, max_iter=max_iter)
    theta, thetaB = res['Theta'], res['ThetaB']
    dtv = (w / 2 * calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1 / (w + 1)
    results_big.loc[ix, 'dtv'] = dtv


results = pd.concat([results_small, results_big])
results.to_csv('Results/results_stop_cond_final.csv', index=False)
