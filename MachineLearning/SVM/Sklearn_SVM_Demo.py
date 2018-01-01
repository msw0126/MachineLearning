#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Sklearn_SVM_Demo.py 
@desc: Sklearn_SVM_Demo
      Sklearn_SVM 译文链接: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10031359
@time: 2018/01/01 
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

print(__doc__)

# 创建40个分离点
np.random.seed(0)

