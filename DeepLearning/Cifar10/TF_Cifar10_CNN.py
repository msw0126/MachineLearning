#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: TF_Cifar10_CNN.py 
@desc: 数据集为Cifar10的CNN实现
@time: 2017/12/21 
"""

# import cifar10.cifar10
# import cifar10.cifar10_input
import tensorflow as tf
import numpy as np
import time

from cifar10 import cifar10
from cifar10 import cifar10_input
from cifar10.cifar10 import maybe_download_and_extract

max_steps = 3000
batch_size = 128
data_dir = 'F:\project\MachineLearning\DeepLearning\Data\cifar10'

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

maybe_download_and_extract()   # 下载解压数据


# 使用distorted_inputs 产生训练需要的数据，包括特征及其对应的label, 该方法做了数据增强
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# cifar10_input.inputs() 生成测试数据
image_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3]) # 3表示有RGB三种通道
label_holder = tf.placeholder(tf.int32, [batch_size])

# 创建卷积层，注意LRN的理解
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64],stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)



