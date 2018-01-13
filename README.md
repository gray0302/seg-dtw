
seg-dtw实现了两种DTW的变体，Segmental DTW和Segmental Local Normalized DTW，分别用Python和Cython进行了实现。
经测试Cython对Python的实现可提速约30倍，编译代码执行python setup.py build_ext --inplace，使用方法参见test.py。
论文参见《Unsupervised spoken keyword spotting via segmental DTW on Gaussian posteriorgrams》、
《Audio keyword extraction by unsupervised word discovery》
该份代码的目的是用于基于实例的语音关键词检索任务。
