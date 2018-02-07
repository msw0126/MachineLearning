#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Learn_Scipy.py
@time: 2018/2/7 13:50
@desc:
"""

import scipy as sc
import numpy as np
from scipy.sparse import coo_matrix

'''
   Scipy学习笔记 
   文档:
     http://python.usyiyi.cn/translate/scipy_lecture_notes/index.html
'''
# 1,稀疏矩阵
# 创建coo矩阵
mtx = coo_matrix((3, 4), dtype=np.int8)
dense = mtx.todense()
print(dense)

