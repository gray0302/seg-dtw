#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2018/1/12 20:39
# @author:Gray

import numpy as np


def distance_mat(s, t, dist):
    N = s.shape[0]
    M = t.shape[0]

    dist_mat = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = dist(s[i], t[j])
    return dist_mat


def diagonal_starts(Nx, Ny, R=1):
    '''
    An auxillairy function based on equations mentioned in the
    "Unsupervised Pattern Discovery..."
    '''

    diagonals = []
    '''
    # no overlap
    lim1 = np.floor((Nx - 1) / (2 * R + 1)).astype(int) + 1
    lim2 = np.floor((Ny - 1) / (2 * R + 1)).astype(int) + 1

    for i in range(0, lim1):
        diagonals.append(((2 * R + 1) * i, 0))

    for j in range(1, lim2):
        diagonals.append((0, (2 * R + 1) * j))
    '''
    # overlap
    lim1 = np.floor((Nx - 1) / R).astype(int) + 1
    lim2 = np.floor((Ny - 1) / R).astype(int) + 1

    for i in range(0, lim1):
        diagonals.append((R * i, 0))

    for j in range(1, lim2):
        diagonals.append((0, R * j))

    return diagonals


def min_around(v):
    m = 0
    for i in range(1, 3):
        if v[i] < v[m]:
            m = i
    return m


def _traceback(mx, my, traceback_mat):
    i = mx
    j = my
    path = [(i, j)]
    while i > 0 and j > 0:
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
    return path


def traceback(start_point, end_point, traceback_mat):
    i = end_point[0]
    j = end_point[1]
    path = [(i, j)]
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
    return path
