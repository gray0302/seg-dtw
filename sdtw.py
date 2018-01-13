#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2018/1/12 20:35
# @author:Gray

from scipy.spatial import distance
import numpy as np
from util import traceback, distance_mat, diagonal_starts, min_around


def segmental_dtw(A, B, R=5, L=50, dist=distance.euclidean):
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
    dist: func
      distance metric
    Returns
    -------
    minimum cost, shortest path
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

    N = len(A)
    M = len(B)

    # TODO: optimize - distance should be added parallely,
    dist_mat = distance_mat(A.astype(np.double),
                            B.astype(np.double), dist)
    diag_starts = diagonal_starts(N, M, R)
    path_fragments = []

    # Computing costs on the diagonals
    for i in range(len(diag_starts)):
        cost_mat, traceback_mat, end_point = dtw_distance(
            dist_mat, diag_starts[i], R)

        # Traceback po optymalnej ścieżce
        path = traceback(diag_starts[i], end_point, traceback_mat)

        # Searching best fragments of paths
        if len(path) >= L:
            d, cost_path = get_dist(dist_mat, path)
            path_fragments.append((d, (diag_starts[i], end_point), cost_path))
    path_fragments = sorted(path_fragments, key=lambda x: x[0])
    if len(path_fragments) == 0:
        return None, None
    else:
        return path_fragments[0][0], path_fragments[0]


def get_dist(dist_mat, path):
    cost_path = np.array([dist_mat[i[0], i[1]] for i in path])
    path_len = (path[0][0] - path[-1][0] + path[0][1] - path[-1][1])
    return np.sum(cost_path) / path_len, cost_path


def dtw_distance(dist_mat, start, R=1):
    costs = np.zeros(3)

    N = dist_mat.shape[0]
    M = dist_mat.shape[1]

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + np.inf
    cost_mat[start[0], start[1]] = 0.

    # Initialize traceback matrix
    traceback_mat = np.zeros((N, M), dtype=np.uint16)

    # Variables for controlling diagonal coordinates
    min_len = min(dist_mat.shape[0], dist_mat.shape[1])

    start_row = start[0] - R if start[0] - R >= 0 else 0
    end_row = start[0] + min_len + R + 1 if start[0] + \
                                            min_len + R < dist_mat.shape[0] else dist_mat.shape[0]
    shift = 0 if start[0] <= start[1] else -R

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

    # Determining the end point
    e0 = end_row - 1 if end_row - 1 >= 0 else 0
    e1 = end_col - 1 if end_col - 1 >= 0 else 0

    return cost_mat, traceback_mat, (e0, e1)
