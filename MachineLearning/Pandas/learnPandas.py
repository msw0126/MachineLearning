#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: learnPandas.py 
@desc:
      中文文档 ：http://python.usyiyi.cn/translate/Pandas_0j2/index.html
@time: 2017/10/06 
"""

import numpy as np
import pandas as pd

# 1,pd.read_csv()  读取csv(逗号分隔)文件到DatFrame
#                 具体说明参考博客: https://www.cnblogs.com/datablog/p/6127000.html
#                   参数说明: sep: str, default ',' 指定分隔符，如果不指定参数，则会尝试使用逗号分隔
#                           delimiter: 定界符,备选分隔符(如果指定该参数，则sep参数失效)