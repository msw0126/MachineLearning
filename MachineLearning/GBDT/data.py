#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: data.py 
@desc: GBDT数据预处理
@time: 2017/12/19 
"""
import theano
import keras
import tensorflow

class DataSet:
    '''
    分类问题默认标签列名称为label，二元分类标签∈{-1, +1}
    回归问题也统一使用label
    '''
    def __init__(self, filename):
        line_cnt = 0
        self.instances = dict()
        self.distinct_valueset = dict()     # just for real value type
        for line in open(filename):
            if line == '\n':
                continue
            fileds = line[:-1].split(',')