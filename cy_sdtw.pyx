#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2018/1/12 21:42
# @author:Gray

import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double log(double theta)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double segmental_dtw(double[:,:] A, double[:,:] B, int R=5, int L=50):
    '''
    Find similarities between two sequences.
    Segmental DTW algorithm extends ide  of Dynamic Time Warping method,
    and looks for the best warping path not only on the main diagonal,
    but also on the other. It facilitates performing not only
    the comparision of the whole sequences, but also discovering similarities
    between subsequences of given sequences A and B.
    Parameters
    ----------
    A: ndarray (n_samples, n_features)
      First sequence
    B: ndarray (n_samples, n_features)
      Second sequence
    R:
      the radius of region
    L:
      Minimal length of path
    Returns
    -------
    minimum cost
    See also
    --------
    dtw
    References
    ----------
    Park A. S. (2006).
    *Unsupervised Pattern Discovery in Speech:
    Applications to Word Acquisition and Speaker Segmentation*
    https://groups.csail.mit.edu/sls/publications/2006/Park_Thesis.pdf
    '''

    cdef int N = A.shape[0]
    cdef int M = B.shape[0]

    # TODO: optimize - distance should be added parallely,
    cdef double[:,:] dist_mat = distance_mat(A, B)
    cdef int[:, :] diag_starts = diagonal_starts(N, M, R)
    cdef list path_fragments = []
    cdef double minimum_cost = 1e8
    cdef int i
    cdef int[:, :] path
    cdef double[:] cost_path
    cdef int path_len
    cdef double d
    cdef double[:, :] cost_mat
    cdef int[:, :] traceback_mat
    cdef int[:] end_point

    # Computing costs on the diagonals
    for i in range(diag_starts.shape[0]):
        cost_mat, traceback_mat, end_point = dtw_distance(
            dist_mat, diag_starts[i], R)

        path = traceback(diag_starts[i], end_point, traceback_mat)

        # Searching best fragments of paths
        if path.shape[0] >= L:
            cost_path = np.array([dist_mat[path[i][0], path[i][1]] for i in range(path.shape[0])])
            path_len = path[0][0] - path[-1][0] + path[0][1] - path[-1][1]
            d = np.sum(cost_path) / path_len
            if minimum_cost < d:
                minimum_cost = d
            path_fragments.append((d, (diag_starts[i], end_point), cost_path))
    return minimum_cost


cdef double[:,:] distance_mat(double[:, :] s, double[:, :] t):
    cdef int N = s.shape[0]
    cdef int M = t.shape[0]

    cdef double[:, :] dist_mat = np.zeros((N, M))
    cdef int i, j
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = distance(s[i], t[j])
    return dist_mat

cdef int[:, :] diagonal_starts(int Nx, int Ny, int R):

    # overlap
    cdef int lim1 = np.floor((Nx - 1) / R).astype(int) + 1
    cdef int lim2 = np.floor((Ny - 1) / R).astype(int) + 1

    cdef int[:, :] diagonals = np.zeros((lim1 + lim2 - 1, 2), dtype=np.int32)
    cdef int i, j

    for i in range(0, lim1):
        diagonals[i] = (R * i, 0)

    for j in range(1, lim2):
        diagonals[lim1 + j - 1] = (0, R*j)

    return diagonals


cdef inline double distance(double[:] s, double[:] t):
    cdef double dot_result = 0.
    cdef int i
    for i in range(s.shape[0]):
        dot_result += s[i]*t[i]
    return -log(dot_result)

cdef tuple dtw_distance(double[:, :] dist_mat, int[:] start, int R=1):
    cdef double[:] costs = np.zeros(3)

    cdef int N = dist_mat.shape[0]
    cdef int M = dist_mat.shape[1]

    # Initialize the cost matrix
    cdef double[:, :] cost_mat = np.zeros((N + 1, M + 1)) + np.inf
    cost_mat[start[0], start[1]] = 0.

    # Initialize traceback matrix
    cdef int[:, :] traceback_mat = np.zeros((N, M), dtype=np.uint16)

    # Variables for controlling diagonal coordinates
    cdef int min_len = np.min(dist_mat.shape[0], dist_mat.shape[1])

    cdef int start_row = start[0] - R if start[0] - R >= 0 else 0
    cdef int end_row = start[0] + min_len + R + 1 if start[0] + \
                                            min_len + R < dist_mat.shape[0] else dist_mat.shape[0]
    cdef int shift = 0 if start[0] <= start[1] else -R
    cdef int start_col = 0
    cdef int end_col = 0
    cdef int i_penalty = 0
    cdef double dist = 0.0
    cdef int i, j

    for i in range(start_row, end_row):
        start_col = max(0, start[1] - R + shift)
        end_col = min(traceback_mat.shape[1], start[1] + R + shift + 1)

        for j in range(start_col, end_col):
            dist = dist_mat[i, j]
            costs[0] = cost_mat[i, j]  # match (0)
            costs[1] = cost_mat[i, j + 1]  # insertion (1)
            costs[2] = cost_mat[i + 1, j]  # deletion (2)
            i_penalty = min_around(costs)
            traceback_mat[i, j] = i_penalty
            cost_mat[i + 1, j + 1] = dist + costs[i_penalty]
        shift += 1

    cdef int[:] end_points = np.zeros(2, dtype=np.int32)
    # Determining the end point
    end_points[0] = end_row - 1 if end_row - 1 >= 0 else 0
    end_points[1] = end_col - 1 if end_col - 1 >= 0 else 0

    return cost_mat, traceback_mat, end_points

cdef inline int min_around(double[:] v):
    cdef int m = 0
    cdef int i
    for i in range(1, v.shape[0]):
        if v[i] < v[m]:
            m = i
    return m

cdef int[:, :] traceback(start_point, end_point, traceback_mat):
    cdef int i = end_point[0]
    cdef int j = end_point[1]
    cdef list path = [(i, j)]
    cdef int tb_type = 0
    while i > start_point[0] or j > start_point[1]:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    return np.asarray(path, dtype=np.int32)
