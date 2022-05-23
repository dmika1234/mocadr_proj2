from funs import *


w = 3
k = 10000
alpha = 0.5
Theta = np.array([[0.375, 0.1, 0.14285714285714285],
                  [0.125, 0.2, 0.2857142857142857],
                  [0.25, 0.3, 0.14285714285714285],
                  [0.25, 0.4, 0.42857142857142855]], )
ThetaB = np.array([0.25, 0.25, 0.25, 0.25])

X = generate_data(Theta, ThetaB, alpha, k, seed=2022)
res = run_em(X, alpha)

theta = res['Theta']
thetaB = res["ThetaB"]

dt = (w/2 * calc_dtv(theta, Theta) + calc_dtv(thetaB, ThetaB)) * 1/(w+1)
print(dt)
