#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2017/11/21 16:29
# @author:Gray

import pyximport
import numpy as np

pyximport.install(setup_args={
    "include_dirs": np.get_include()})
import cy_slndtw
from scipy.io import wavfile
import python_speech_features
import time


def feat_extract(filename):
    sr, y = wavfile.read(filename)
    return python_speech_features.mfcc(y, sr)


if __name__ == '__main__':
    query_file = '/tmp/temp_1.wav'
    search_file = '/tmp/temp_2.wav'
    query = feat_extract(query_file)
    search = feat_extract(search_file)
    st_time = time.time()
    print(cy_slndtw.sln_dtw(query, search))
    # print(util.sln_dtw(query, search, dist=lambda x, y: -np.log(np.dot(x, y)))[0])
    print(time.time() - st_time)
