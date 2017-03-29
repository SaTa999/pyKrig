#!/usr/bin/env python
from math import ceil, sqrt
import numpy as np
from functools import partial
from pyKrig.utilities import pairwise_distance, mmcriterion


def latin_hypercube(nsample, ndv):
    """
    create sample points for latin hypercube sampling
    :param nsample: number of samples
    :param ndv: number of design variables
    """
    min_level = (1 / nsample) * 0.5
    max_level = 1 - min_level
    levels = np.linspace(min_level, max_level, nsample)
    lhs = np.empty((nsample, ndv))
    index_levels = np.arange(nsample)
    for j in range(ndv):
        order = np.random.permutation(index_levels)
        lhs[:, j] = levels[order]
    return lhs


def perturbate(X):
    """
    randomly choose a pair of sampling points, and interchange the values of a randomly chosen design variable
    """
    ns, ndv = X.shape
    is1 = np.random.randint(ns)
    is2 = np.random.randint(ns)
    idv = np.random.randint(ndv)
    X[is1, idv], X[is2, idv] = X[is2, idv], X[is1, idv]


def optimize_lhs(X, criterion_func):
    """
    optimize a latin hypercube via simulated annealing
    :param X: a latin hypercube
    :param criterion_func: the function used to evaluate the latin hypercube
    """
    # initialize
    phi = criterion_func(X)
    phi_best = phi
    Xbest = np.array(X, copy=True)
    Xtry = np.empty_like(X)

    # calculate initial temperature by sampling the average change of criterion
    n_test = 30
    avg_delta_phi = 0
    cnt = 0
    for _ in range(n_test):
        Xtry[:] = X
        perturbate(Xtry)
        phi_try = criterion_func(Xtry)
        delta_phi = phi_try - phi
        if delta_phi > 0:
            cnt += 1
            avg_delta_phi += delta_phi
    if cnt == 0:
        temp_init = 1.
    else:
        avg_delta_phi /= cnt
        temp_init = - avg_delta_phi / np.log(0.99)
    temp_fin = temp_init * 1e-6
    cool_rate = 0.95
    max_perturbation = ceil(sqrt(X.shape[1]))
    temp = temp_init

    # optimize lhs via simmulated annealing
    while temp > temp_fin:
        Xtry[:] = X
        for _ in range(max_perturbation):
            perturbate(Xtry)
            phi_try = criterion_func(Xtry)
            if phi_try < phi_best:
                Xbest[:] = Xtry
                phi_best = phi_try
            delta_phi = phi_try - phi
            if delta_phi < 0:
                X[:] = Xtry
                phi = phi_try
                break
            elif np.exp(- delta_phi / temp) > np.random.rand():
                X[:] = Xtry
                phi = phi_try
        temp *= cool_rate
    X[:] = Xbest


def optimal_latin_hypercube(nsample, ndv, metric="euclidean"):
    """
    create an optimal lhs using Morris & Mitchel's method (1995)
    :param nsample: number of sample points
    :param ndv: number of design variables
    :param metric: metric used to calculate the MM criterion; use "manhattan" or "euclidean"
    """
    X = latin_hypercube(nsample, ndv)
    Xbest = np.array(X, copy=True)
    distbest = pairwise_distance(Xbest, metric)
    Xtry = np.empty_like(X)
    disttry = np.empty_like(distbest)
    for p in (1, 2, 5, 10, 20, 50, 100):
        cfunc = partial(mmcriterion, p=p, metric=metric)
        Xtry[:] = X
        optimize_lhs(Xtry, cfunc)
        disttry[:] = pairwise_distance(Xtry, metric)
        for dtry, dbest in zip(disttry, distbest):
            if dtry > dbest:
                Xbest[:] = Xtry
                distbest[:] = disttry
                break
            elif dtry < dbest:
                break
            else:
                continue
    return Xbest
