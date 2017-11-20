#-*- coding:utf-8 _*-
"""
@Author:charlesXu
@File: process_data.py
@Desc: Tensorflow做完形填空 模型实现
@Time: 2017/11/20
"""

import tensorflow as tf
import pickle
import numpy as np
import ast

from collections import defaultdict
from tensorflow.contrib.layers.python.layers import regularizers

train_data = 'train.vec'
valid_data = 'valid.vec'

word2idx, content_length, question_length, vocab_size = pickle.load(open('vocab.data', 'rb'))
print(content_length, question_length, vocab_size)

batch_size = 64

train_file = open(train_data)

def get_next_batch():
    X = []
    Q = []
    A = []
    for i in range(batch_size):
        for line in train_file:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
            break

    if len(X) == batch_size:
        return X, Q, A
    else:
        train_file.seek(0)
        return get_next_batch()

def get_text_batch():
    with open('valid_data') as f:
        X = []
        Q = []
        A = []
        for line in f:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
        return X, Q, A

X = tf.placeholder(tf.int32, [batch_size, content_length])  # 英文材料
Q = tf.placeholder(tf.int32, [batch_size, question_length])  # 问题
A = tf.placeholder(tf.int32, [batch_size])                  # 答案

# Drop out
keep_prob = tf.placeholder(tf.float32)

def glimpse(weights, bias, encodings, inputs):
    weights = tf.nn.dropout(weights, keep_prob)
    inputs = tf.nn.dropout(inputs, keep_prob)
    attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
    attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
    attention = tf.nn.softmax(tf.squeeze(attention, -1)) # tf.squeeze() 从tensor中删除所有大小是1的维度。
    return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)

def neural_attention(embedding_dim=384, encoding_dim=128):
    embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_dim], stddev=0.22), dtype=tf.float32)
    regularizers.apply_regularization(regularizers.l2_regularizer(1e-4), [embeddings])

    with tf.variable_scope('encode'):
        with tf.variable_scope('X'):
            X_lens = tf.reduce_sum(tf.sign(tf.abs(X)), 1)
            embedded_X = tf.nn.embedding_lookup(embeddings, X)
            encoded_X = tf.nn.dropout(embedded_X, keep_prob)
            gru_cell = tf.nn.rnn_cell.GRUCell(embedding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, embedded_X, sequence_length=X_lens, dtype=tf.float32, swap_memory=True)
            encoded_X = tf.concat(2, outputs)
        with tf.variable_scope('Q'):
            Q_lens = tf.reduce_sum(tf.sign(tf.abs(Q)), 1)
            embedded_Q = tf.nn.embedding_lookup(embeddings, Q)
            encoded_Q = tf.nn.dropout(embedded_Q, keep_prob)
            gru_cell = tf.nn.rnn_cell.GRUCell(encoding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_Q,
                                                                     sequence_length=Q_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            encoded_Q = tf.concat(2, outputs)

    W_q = tf.Variable(tf.random_normal([2 * encoding_dim, 4 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_q = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    W_d = tf.Variable(tf.random_normal([2 * encoding_dim, 6 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_d = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    g_q = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)
    g_d = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)

    with tf.variable_scope('attend') as scope:
        infer_gru = tf.nn.rnn_cell.GRUCell(4*encoding_dim)
        infer_state = infer_gru.zero_state(batch_size, tf.float32)
        for iter_step in range(8):
            if iter_step > 0:
                scope.reuse_variables()

