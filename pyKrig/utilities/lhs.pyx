#!python
#cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport pow, abs, sqrt


# define a function pointer to a metric
ctypedef double (*metric_ptr)(double[:, ::1], np.intp_t, np.intp_t)


cdef inline double _euclidean_dist(double[:, ::1] X, np.intp_t i1, np.intp_t i2):
    """
    calculate the squared euclidean distance between points X[i1] and X[i2]
    """
    cdef double tmp
    cdef double d = 0
    cdef int ndv = X.shape[1]
    cdef np.intp_t k
    for k in range(ndv):
        tmp = X[i1, k] - X[i2, k]
        d += tmp * tmp
    return d


cdef inline double _manhattan_dist(double[:, ::1] X, np.intp_t i1, np.intp_t i2):
    """
    calculate the manhattan distance between points X[i1] and X[i2]
    """
    cdef double d = 0
    cdef int ndv = X.shape[1]
    cdef np.intp_t k
    for k in range(ndv):
        d += abs(X[i1, k] - X[i2, k])
    return d


# define a function pointer to a metric
ctypedef double (*pow_ptr)(double, int)


cdef inline double _fast_pow1(double a, int b):
    """
    calculate a**b using exponentiation by squaring
    use when metric is manhattan
    """
    cdef double res = 1
    while b:
        if b & 1:
            res *= a
        b >>= 1
        a *= a
    return res


cdef inline double _fast_pow2(double a, int b):
    """
    calculate a**(b/2) using exponentiation by squaring
    use when metric is euclidean
    """
    cdef:
        double res = 1
    if b & 1:
        res *= sqrt(a)
    b >>= 1
    res *= _fast_pow1(a, b)
    return res


def mmcriterion(double[:, ::1] design, int p, str metric="euclidean"):
    """
    calculate the Morris-Mitchell criterion of an experimental design
    :param design: 2d-array representing an experimental design
                   each row corresponds to a sample point,
                   and each column corresponds to a design variable
    :param p: positive integer; try values such as (1, 2, 5, 10, 20, 50, and 100)
    """
    cdef int ns = design.shape[0]
    cdef int ndv = design.shape[1]
    cdef np.intp_t i, j
    cdef double phi = 0
    cdef double dist
    cdef double inv_p
    cdef metric_ptr dist_func
    cdef pow_ptr fast_pow
    if metric == "euclidean":
        dist_func = &_euclidean_dist
        fast_pow = &_fast_pow1
    elif metric == "manhattan":
        dist_func = &_manhattan_dist
        fast_pow = &_fast_pow2
    else:
        raise ValueError("metric must be euclidean or manhattan")
    for i in range(ns):
        for j in range(i+1, ns):
            dist = dist_func(design, i, j)
            phi += fast_pow(1/dist, p)
    inv_p = 1
    inv_p /= p
    phi = pow(phi, inv_p)
    return phi


def pairwise_distance(double[:, ::1] design, str metric="euclidean"):
    """
    calculate the pairwise distance between points in an experimental design
    :param design: 2d-array representing an experimental design
                   each row corresponds to a sample point,
                   and each column corresponds to a design variable
    """
    cdef int ns = design.shape[0]
    cdef int ndv = design.shape[1]
    cdef int npairs = ns * (ns - 1) // 2
    cdef np.intp_t i, j
    cdef np.intp_t k = 0
    cdef double[:] dist = np.empty(npairs, dtype=np.float64)
    cdef metric_ptr dist_func
    if metric == "euclidean":
        dist_func = &_euclidean_dist
    elif metric == "manhattan":
        dist_func = &_manhattan_dist
    else:
        raise ValueError("metric must be euclidean or manhattan")
    k = 0
    for i in range(ns):
        for j in range(i+1, ns): 
            dist[k] = dist_func(design, i, j)
            k += 1
    return np.sort(np.asarray(dist))
