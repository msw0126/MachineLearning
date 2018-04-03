#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Model.py
@time: 2018/4/3 17:50
@desc: BiLSTM + CRF 模型层
"""

import numpy as np
import os, time, sys

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

from Data import pad_sequences, batch_yield
from Utils import get_logger
from Eval import conlleval

if __name__ == '__main__':
    pass