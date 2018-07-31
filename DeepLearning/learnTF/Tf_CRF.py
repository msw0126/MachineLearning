# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     Tf_CRF
   Description :  tensorflow有关crf的学习笔记
   Author :       charl
   date：          2018/7/31
-------------------------------------------------
   Change Activity:
                   2018/7/31:
-------------------------------------------------
"""

import numpy as np
import tensorflow as tf

#data settings
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# 5 tags
# random features
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# 序列的长度
sequence_lenths = np.full(num_examples, num_words - 1, dtype=np.int32)
