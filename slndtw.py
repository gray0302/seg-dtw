#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2018/1/12 20:38
# @author:Gray

from scipy.spatial import distance
import numpy as np
from util import _traceback, distance_mat, min_around


def sln_dtw(query, search, dist=distance.euclidean):
    '''
    Segmental Locally Normalized DTW
    :param query: ndarray (n_samples, n_features)
       keyword template
    :param search: ndarray (n_samples, n_features)
       indexed utterance
    :return:
    minimum cost, shortest path
    '''
    N = len(query)
    M = len(search)
    traceback_mat = np.zeros((N, M), dtype=np.uint16)
    dist_mat = distance_mat(query, search, dist)
    acc, l = initialize(dist_mat)
    costs = np.zeros(3)
    for i in range(1, N):
        for j in range(1, M):
            dist = dist_mat[i, j]
            costs[0] = (acc[i - 1, j - 1] + dist) / (l[i - 1, j - 1] + 1)
            costs[1] = (acc[i - 1, j] + dist) / (l[i - 1, j] + 1)
            costs[2] = (acc[i, j - 1] + dist) / (l[i, j - 1] + 1)
            i_penalty = min_around(costs)
            traceback_mat[i, j] = i_penalty
            if i_penalty == 0:
                acc[i, j] = acc[i - 1, j - 1] + dist
                l[i, j] = l[i - 1, j - 1] + 1
            elif i_penalty == 1:
                acc[i, j] = acc[i - 1, j] + dist
                l[i, j] = l[i - 1, j] + 1
            else:
                acc[i, j] = acc[i, j - 1] + dist
                l[i, j] = l[i, j - 1] + 1
    min_cost = 1e8
    end_j = 0
    for j in range(M):
        cost = acc[N - 1, j] / l[N - 1, j]
        if cost < min_cost:
            min_cost = cost
            end_j = j
    path = _traceback(N - 1, end_j, traceback_mat)
    return min_cost, path


def initialize(dist_mat):
    N, M = dist_mat.shape
    acc = np.zeros((N, M))
    l = np.zeros((N, M))
    for i in range(N):
        acc[i, 0] = np.sum(dist_mat[:i + 1, 0])
        l[i, 0] = i + 1
    for j in range(M):
        acc[0, j] = dist_mat[0, j]
        l[0, j] = 1
    return acc, l
