import numpy as np
import random as rng
from functools import partial
from time import time
import copy


def generate_data(Theta, ThetaB, alfa, k, seed: int = 1):
    w = Theta.shape[1]
    rng.seed(seed)
    res = np.zeros((k, w))
    for i in np.arange(0, k):
        z = np.random.choice(np.arange(0, 2), p=[1 - alfa, alfa])
        if z == 0:
            res[i, :] = np.random.choice(np.arange(1, 5), p=ThetaB, size=w)
        if z == 1:
            for j in np.arange(0, w):
                res[i, j] = np.random.choice(np.arange(1, 5), p=Theta[:, j])

    return np.array(res, dtype='int32')


def generate_thetas(w: int, seed: int = 1):
    np.random.seed(seed)
    thetaB = np.random.random(4)
    thetaB = thetaB / thetaB.sum()
    theta = np.random.random(4 * w)
    theta.shape = (4, w)
    theta = theta / theta.sum(axis=0, keepdims=1)
    return theta, thetaB


def calc_dtv(p, q):
    diff = p - q
    res = 1 / 2 * np.sum(np.abs(diff))
    return res


def calc_pii(vec, theta, thetaB, alpha: float) -> np.ndarray:
    ix = vec - 1
    mask = (ix[:, None] == np.arange(4)).T
    prod_theta = np.prod(theta[mask])
    prod_thetaB = np.prod(np.take(thetaB, ix))
    res = (alpha * prod_theta) / (alpha * prod_theta + (1 - alpha) * prod_thetaB)
    return res


def run_em_stop_cond(data, alpha: int = None, max_iter: int = 700, seed: int = 1):
    np.random.seed(seed)
    data = np.array(data, dtype='int32')
    w = data.shape[1]
    thetaB = np.random.random(4) / 4
    thetaB[0] = 1 - np.sum(thetaB[1:])
    theta = np.random.random(4 * w) / (4 * w)
    theta.shape = (4, w)
    theta[0, :] = 1 - theta[1:, :].sum(axis=0)
    rep = 0
    theta_next = copy.deepcopy(theta)
    thetaB_next = copy.deepcopy(thetaB)
    norm = np.zeros((max_iter,))
    normB = np.zeros((max_iter,))


    while rep < max_iter:
        # Calculating pi
        calc_pi_mapfunc = partial(calc_pii, theta=theta, thetaB=thetaB, alpha=alpha)
        pi = np.array(list(map(calc_pi_mapfunc, data)))
        for m in np.arange(4):
            # Derivative for Theta
            numerator = (np.array((data == m + 1), dtype='float32').T * pi).T
            theta_next[m, :] = np.apply_along_axis(np.sum, axis=0, arr=numerator) / np.sum(pi)
            # Derivative for ThetaB
            Al = np.apply_along_axis(np.sum, axis=1, arr=(data == m + 1))
            thetaB_next[m] = (np.sum((1 - pi) * Al)) / (w * np.sum(1 - pi))
        diff = theta_next - theta
        norm[rep] = np.sqrt(((diff ** 2).sum()) / 4 * w)
        diffB = thetaB_next - thetaB
        normB[rep] = np.sqrt(((diffB ** 2).sum()) / 4)
        theta = copy.deepcopy(theta_next)
        thetaB = copy.deepcopy(thetaB_next)
        rep += 1

    return norm, normB


def run_em(data, alpha: int = None, max_iter: int = 600, min_diff: float = 1e-10, seed: int = 1) -> dict:
    tic = time()
    np.random.seed(seed)
    data = np.array(data, dtype='int32')
    w = data.shape[1]
    theta, thetaB = generate_thetas(w, seed)
    rep = 0
    theta_next = copy.deepcopy(theta)
    thetaB_next = copy.deepcopy(thetaB)
    norm = min_diff + 1
    normB = min_diff + 1

    results = dict({"max_iter": max_iter, "min_diff": min_diff,
                    "alpha": alpha, "Theta": theta, "ThetaB": thetaB,
                    "last_diff": None, "last_diffB": None, "iterations": 0, "time_spent": None})

    while (rep < max_iter) & ((min_diff < norm) | (min_diff < normB)):
        # Calculating pi
        calc_pi_mapfunc = partial(calc_pii, theta=theta, thetaB=thetaB, alpha=alpha)
        pi = np.array(list(map(calc_pi_mapfunc, data)))
        for m in np.arange(4):
            # Derivative for Theta
            numerator = (np.array((data == m + 1), dtype='float32').T * pi).T
            theta_next[m, :] = np.apply_along_axis(np.sum, axis=0, arr=numerator) / np.sum(pi)
            # Derivative for ThetaB
            Al = np.apply_along_axis(np.sum, axis=1, arr=(data == m + 1))
            thetaB_next[m] = (np.sum((1 - pi) * Al)) / (w * np.sum(1 - pi))
        diff = theta_next - theta
        norm = np.sqrt(((diff ** 2).sum()) / 4 * w)
        diffB = thetaB_next - thetaB
        normB = np.sqrt(((diffB ** 2).sum()) / 4)
        theta = copy.deepcopy(theta_next)
        thetaB = copy.deepcopy(thetaB_next)
        rep += 1
    toc = time()

    results['Theta'] = theta
    results['ThetaB'] = thetaB
    results['alpha'] = alpha
    results['last_diff'] = norm
    results['last_diffB'] = normB
    results['iterations'] = rep
    results['time_spent'] = toc - tic

    return results


def run_em2(data, alpha: int = None, max_iter: int = 100, seed: int = 1) -> dict:
    tic = time()
    np.random.seed(seed)
    data = np.array(data, dtype='int32')
    w = data.shape[1]
    theta, thetaB = generate_thetas(w, seed)
    rep = 0

    results = dict({"max_iter": max_iter,
                    "alpha": alpha, "Theta": theta, "ThetaB": thetaB,
                    "last_diff": None, "last_diffB": None, "iterations": 0, "time_spent": None})

    while rep < max_iter:
        # Calculating pi
        calc_pi_mapfunc = partial(calc_pii, theta=theta, thetaB=thetaB, alpha=alpha)
        pi = np.array(list(map(calc_pi_mapfunc, data)))
        for m in np.arange(4):
            # Derivative for Theta
            numerator = (np.array((data == m + 1), dtype='float32').T * pi).T
            theta[m, :] = np.apply_along_axis(np.sum, axis=0, arr=numerator) / np.sum(pi)
            # Derivative for ThetaB
            Al = np.apply_along_axis(np.sum, axis=1, arr=(data == m + 1))
            thetaB[m] = (np.sum((1 - pi) * Al)) / (w * np.sum(1 - pi))
        rep += 1

    toc = time()

    results['Theta'] = theta
    results['ThetaB'] = thetaB
    results['alpha'] = alpha
    results['iterations'] = rep
    results['time_spent'] = toc - tic

    return results
