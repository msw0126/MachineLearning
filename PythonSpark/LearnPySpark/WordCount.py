#-*- coding:utf-8 _*-
"""
@author:charlesXu
@file: WordCount.py
@desc: pyspark的HelloWorld
@time: 2017/12/27
"""

from __future__ import print_function

import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf

if __name__=='__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: wordcount <file>", file=sys.stderr)
    #     exit(-1)

    # conf = SparkConf.setMaster('local').setAppName('WordCount')
    # sc = SparkContext(conf)
    # sc.textFile('E:\py_workspace\MachineLearning\PythonSpark\Data\wordcount.txt')
    # lines = sc.
    spark = SparkSession.builder\
                        .appName('pyWordCount')\
                        .getOrCreate()
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' '))\
                  .map(lambda x: (x, 1))\
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print('%s: %i' % (word, count))

    spark.stop()