#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: XgBoost_Intro.py 
@desc: XgBoost学习笔记，入门
官网文档：http://xgboost.readthedocs.io/en/latest/get_started/index.html
@time: 2017/12/18 
"""

import xgboost as xgb
import numpy as np

# XgBoost的基本使用
# 自定义损失函数的梯度和二阶导


def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

if __name__=='__main__':
    # 读取数据
    data_train = xgb.DMatrix('F:\project\MachineLearning\MachineLearning\Data\XgBoost\\agaricus_train.txt')
    data_test = xgb.DMatrix('F:\project\MachineLearning\MachineLearning\Data\XgBoost\\agaricus_test.txt')
    # print(data_train)
    # print(type(data_train))

    # 设置参数
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_around = 3
    bst = xgb.train(param, data_train, num_boost_round=n_around, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)

    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误总数: \t%4d' % error)
    print('错误率: \t%.5f%%' % (100* error_rate))