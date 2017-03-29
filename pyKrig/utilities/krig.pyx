#!python
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as lp
cimport scipy.linalg.cython_blas as bl
cimport cython
from libc.math cimport exp


def pdist(double[:, ::1] X, double[:] b, double reg=0.):
    """
    calculate the pairwise distance between each sample point
    """
    cdef int M = X.shape[0]
    cdef int N = X.shape[1]
    cdef int i, j
    cdef double tmp, d
    cdef double regp1 = reg + 1.
    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        D[i, i] = regp1
        for j in range(i+1, M):
            d = 0.
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d -= tmp * tmp * b[k]
            D[i, j] = exp(d)
    return np.asarray(D)


def chol_fact(double[:,::1] A, char UPLO='L'):
    """
    cholesky decomposition A = L * L^t 
    """
    cdef double[::1, :] AT = A.T
    cdef int N = A.shape[0]
    cdef int info
    lp.dpotrf(&UPLO, &N, &AT[0,0], &N, &info)
    if info > 0:
        raise np.linalg.LinAlgError("%d-th leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potrf' % -info)


def chol_solve(double[:,::1] A, double[:, ::1] B, char UPLO='L'):
    cdef double[::1, :] AT = A.T
    cdef double[::1, :] BT = B.T
    cdef int N = A.shape[0]
    cdef int NRHS = B.shape[0]
    cdef int info
    lp.dpotrs(&UPLO, &N, &NRHS, &AT[0,0], &N, &BT[0,0], &N, &info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potrs' % -info)


def chol_inv(double[:,::1] A, char UPLO='L'):
    cdef double[::1, :] AT = A.T
    cdef int N = A.shape[0]
    cdef int info
    lp.dpotri(&UPLO, &N, &AT[0,0], &N, &info)
    if info > 0:
        raise np.linalg.LinAlgError("%d-th leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potri' % -info)


def gll(double[:] nf_ir, double[:, ::1] normx, double[:,::1]rm, double[:, ::1] ir, double sigma):
    cdef Py_ssize_t M = normx.shape[0]
    cdef Py_ssize_t N = normx.shape[1]
    cdef double tmp, rmarm, invsig = 1.0 / sigma
    cdef double[:] grad = np.zeros(N, dtype=np.float64)
    cdef int i, j
    for i in range(M):
        for j in range(i+1, M):
            rmarm = rm[i,j] * (nf_ir[i] * nf_ir[j] * invsig - ir[i, j])
            for k in range(N):
                tmp = normx[i, k] - normx[j, k]
                grad[k] -= tmp * tmp * rmarm
    return np.asarray(grad)


def gll_reg(double[:] nf_ir, double[:, ::1] normx, double[:,::1]rm, double[:, ::1] ir, double sigma):
    cdef Py_ssize_t M = normx.shape[0]
    cdef Py_ssize_t N = normx.shape[1]
    cdef int i, j, k
    cdef double tmp, rmarm, invsig = 1.0 / sigma
    cdef double[:] grad = np.zeros(N+1, dtype=np.float64)
    for i in range(M):
        for j in range(i+1, M):
            rmarm = rm[i,j] * (nf_ir[i] * nf_ir[j] * invsig - ir[i, j])
            for k in range(N):
                tmp = normx[i, k] - normx[j, k]
                grad[k+1] -= tmp * tmp * rmarm
        rmarm = rm[i,i] * (nf_ir[i] * nf_ir[i] * invsig - ir[i, i])
        grad[0] += rmarm
    grad[0] *= 0.5
    return np.asarray(grad)


def symdot(double[:,::1] A, double[:] B, char side='R', char uplo='L'):
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef double[::1, :] AT = A.T
    cdef int m = 1
    cdef int n = A.shape[0]
    cdef int ldA = A.shape[1]
    cdef int ldB = 1
    cdef int ldC = 1
    cdef double[:] C = np.empty(A.shape[0], dtype=np.float64)
    
    bl.dsymm(&side, &uplo, &m, &n, &alpha, &AT[0,0], &ldA, &B[0], &ldB, &beta, &C[0], &ldC)
    return np.asarray(C)